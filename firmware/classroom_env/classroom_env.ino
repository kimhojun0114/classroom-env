/*
 * 교실 학습 환경 조절기 — ESP32 펌웨어
 * 20713 김호준 · 인공지능 융합 프로젝트
 *
 * 하는 일
 *  1) 소리(MAX4466)는 50ms 창 단위로 계속 측정(8kHz 샘플링 + FFT로 주파수 특징 계산), CO2(UART·PWM 두 방식)·온습도·조도는 1초마다 측정
 *  2) 10초마다 평균을 내서 하나의 기록으로 만들고 Firebase 에 업로드
 *  3) 최근 10분 기록으로 특징 8개를 계산해 학습 상태(최적/졸림/산만)를 판단
 *     - USE_MLP 0 : 임시 규칙(데이터 수집 기간용)
 *     - USE_MLP 1 : 학습된 인공신경망(model_weights.h)
 *  4) 신호등 LED · LCD · 서보모터(창문 모형) 제어, 웹앱의 원격 설정(/control) 반영
 */
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiMulti.h>
#include <WebServer.h>
#include <DNSServer.h>
#include <Preferences.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <time.h>
#include <math.h>
#include <BH1750.h>
#include <LiquidCrystal_I2C.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>
#include "config.h"

#ifndef USE_MLP
#define USE_MLP 0
#endif
#ifndef REED_COUNT
#define REED_COUNT 0
#endif

#if SHT_MODEL == 40
#include <Adafruit_SHT4x.h>
Adafruit_SHT4x sht;
#else
#include <Adafruit_SHT31.h>
Adafruit_SHT31 sht;
#endif

#if USE_MLP
#include "model_weights.h"
#endif

// ───────── 핀 배치 ─────────
const int PIN_MIC     = 34;   // MAX4466 OUT (ADC1, 와이파이와 같이 사용 가능)
const int PIN_LED_R   = 25;
const int PIN_LED_Y   = 26;
const int PIN_LED_G   = 27;
const int PIN_SERVO   = 23;   // 서보모터 신호선(주황/노랑)
const int SERVO_CLOSED = 0;   // 창문 닫힘 각도
const int SERVO_OPEN   = 90;  // 창문 열림 각도 (모형에 맞게 조정)
const int PIN_CO2_RX  = 16;   // ESP32 RX2 ← MH-Z19C TX
const int PIN_CO2_TX  = 17;   // ESP32 TX2 → MH-Z19C RX
const int PIN_CO2_PWM = 4;    // MH-Z19C PWM 출력 (UART 값 검증용)
// I2C: SDA 21, SCL 22 (LCD, BH1750, SHT 공유)
const int REED_PINS[8] = {13, 14, 18, 19, 32, 33, 5, 15};  // D4는 CO2 PWM이 사용

// ───────── 시간 설정 ─────────
const uint32_t ENV_READ_MS  = 1000;    // CO2·온습도·조도 측정 주기
const uint32_t RECORD_MS    = 10000;   // 기록·업로드 주기
const uint32_t CONTROL_MS   = 10000;   // 원격 설정 읽기 주기
const uint32_t WARMUP_S     = 180;     // MH-Z19C 예열 시간
const int      BUF_N        = 60;      // 10초 × 60 = 최근 10분
const int      PENDING_MAX  = 90;      // 업로드 실패 시 보관할 기록 수(15분)

// ───────── 소리 급증(말소리·충격음) 감지 기준 ─────────
const float SPIKE_RATIO   = 3.0f;   // 배경 대비 몇 배 이상이면 급증
const float SPIKE_MIN_ADC = 400.0f; // 배경보다 최소 이만큼(ADC) 커야 급증
const int   SPIKE_HOLD    = 3;      // 50ms 창이 연속 3번(150ms) 넘어야 소리로 인정
                                    // → 키보드·딸깍 같은 짧은 소리와 전기 잡음은 제외, 말소리만 셈

// ───────── 원격 설정 (웹앱 /control 에서 변경 가능) ─────────
struct Control {
  float co2Drowsy = 2000;   // 졸음 임계값(ppm) — 데이터로 찾은 뒤 웹앱에서 수정
  float co2Clear  = 1000;   // 환기 알림 해제 기준(ppm)
  float etaMin    = 10;     // 임계값 도달 예측 시간이 이 값(분) 이하이면 미리 알림
  float luxLow    = 100;    // 이보다 어두우면 '조명 부족' (학교보건법 300lx는 웹앱에서 조정)
  float tMin = 18, tMax = 28;   // 온도 기준(℃)
  float hMin = 30, hMax = 80;   // 습도 기준(%)
  float loudLvl   = 500;    // 최근 5초 평균 소리 크기(진폭)가 이 이상이면 '계속 시끄러움' → 산만
  float noise30   = 6;      // 최근 30초 급증 횟수가 이 이상이면 '산만' (2회 이하로 줄면 해제)
  String winMode  = "auto"; // 창문(서보): auto / on(열기) / off(닫기)
} ctl;

// ───────── 상태 ─────────
enum State { ST_OPTIMAL = 0, ST_DROWSY = 1, ST_DISTRACTED = 2 };
const char* STATE_KO[] = {"최적", "졸림", "산만"};

struct Rec {
  time_t ts;
  float sndMean, sndStd, sndMax;
  int   sndSpikes;
  float co2, co2pwm, temp, hum, lux;
  float sndSp, sndFlat, sndCent;    // 주파수 특징
};
Rec buf[BUF_N];
int bufCount = 0, bufHead = 0;   // bufHead = 다음에 쓸 위치

struct Features {           // 인공신경망 입력 순서와 동일하게 유지할 것
  float sndMean, sndStd, sndSpikes, co2Mean, co2Slope, lux, temp, hum;
};

// ───────── 장치 ─────────
BH1750 lightMeter;
LiquidCrystal_I2C lcd(LCD_ADDR, 16, 2);
HardwareSerial co2Serial(2);
bool shtOk = false, luxOk = false;

// ───────── 소리 측정 (별도 태스크) ─────────
portMUX_TYPE sndMux = portMUX_INITIALIZER_UNLOCKED;
float sndSum = 0, sndSq = 0, sndMax = 0;
int   sndN = 0, sndSpikes = 0;
// 실시간 소음 판정용: 최근 급증 시각(ms) 기록
const int SPK_RING = 32;
uint32_t spikeT[SPK_RING] = {0};
int spikeHead = 0;
// 지속 소음 판정용: 1초 평균 진폭을 최근 10초치 보관
const int SEC_RING = 10;
float secLvl[SEC_RING] = {0};
int secHead = 0, secFilled = 0;

// ───────── 주파수 분석 (FFT) ─────────
// 8kHz로 256개(32ms)를 모아 FFT → 소리 종류를 구분하는 특징 3개
//  · snd_sp   : 말소리 대역(300~3000Hz) 에너지 비율 (말소리↑)
//  · snd_flat : 스펙트럼 평탄도 0~1 (백색소음·환풍기↑, 말소리·음악↓)
//  · snd_cent : 스펙트럼 중심 Hz (키보드·딸깍 같은 높은 소리↑)
// 10초 기록에는 '소리 에너지로 가중 평균'해서 저장 → 큰 소리가 날 때의 특징이 주로 반영됨
const int      FFT_N     = 256;
const uint32_t SAMPLE_US = 125;          // 8000Hz
float fRe[FFT_N], fIm[FFT_N], hannW[FFT_N];
double specW = 0, specSp = 0, specFl = 0, specCt = 0;   // sndMux 로 보호

void fftInPlace(float* re, float* im, int n) {
  for (int i = 1, j = 0; i < n; i++) {                 // 비트 뒤집기 정렬
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { float t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (int len = 2; len <= n; len <<= 1) {             // 나비 연산
    float ang = -2.0f * (float)M_PI / len, wr = cosf(ang), wi = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float cr = 1, ci = 0;
      for (int k = 0; k < len / 2; k++) {
        int a = i + k, b = i + k + len / 2;
        float vr = re[b] * cr - im[b] * ci, vi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - vr; im[b] = im[a] - vi;
        re[a] += vr;        im[a] += vi;
        float t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

// x[0..FFT_N) 원신호, fs 샘플링 주파수 → 특징 3개. 소리가 거의 없으면 false
bool specFeatures(float* x, float fs, float& sp, float& fl, float& ct) {
  float mean = 0;
  for (int i = 0; i < FFT_N; i++) mean += x[i];
  mean /= FFT_N;
  for (int i = 0; i < FFT_N; i++) { fRe[i] = (x[i] - mean) * hannW[i]; fIm[i] = 0; }
  fftInPlace(fRe, fIm, FFT_N);
  float binHz = fs / FFT_N;
  double tot = 0, spE = 0, wsum = 0, logSum = 0; int m = 0;
  for (int k = 1; k < FFT_N / 2; k++) {
    float f = k * binHz;
    if (f < 60) continue;                               // 전원 험·직류 성분 제외
    double P = (double)fRe[k] * fRe[k] + (double)fIm[k] * fIm[k];
    tot += P; wsum += P * f;
    if (f >= 300 && f <= 3000) spE += P;
    logSum += log(P + 1e-6); m++;
  }
  if (tot < 1e-3 || m == 0) return false;
  sp = spE / tot;
  ct = wsum / tot;
  fl = exp(logSum / m) / (tot / m);                     // 기하평균 / 산술평균
  return true;
}

void soundTask(void*) {
  for (int i = 0; i < FFT_N; i++) hannW[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (FFT_N - 1));
  float baseline = -1;
  bool inSpike = false;
  int  overRun = 0, heldRun = 0, secN = 0;
  float secSum = 0;
  for (;;) {
    int mn = 4095, mx = 0, ns = 0;
    static float raw[FFT_N];
    uint32_t t0 = micros(), next = t0;
    while (micros() - t0 < 50000) {            // 50ms 동안 8kHz로 측정, 최댓값-최솟값 = 진폭
      while ((int32_t)(micros() - next) < 0) {}
      next += SAMPLE_US;
      int v = analogRead(PIN_MIC);
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      if (ns < FFT_N) raw[ns] = v;
      ns++;
    }
    float p2p = mx - mn;
    float sp = 0, fl = 0, ct = 0;
    bool specOk = ns >= FFT_N && specFeatures(raw, ns * 20.0f, sp, fl, ct);   // 실제 샘플링 주파수 = ns / 0.05초
    if (baseline < 0) baseline = p2p;
    bool over = (p2p > baseline * SPIKE_RATIO) && (p2p - baseline > SPIKE_MIN_ADC);
    overRun = over ? overRun + 1 : 0;
    bool spikeNow = overRun >= SPIKE_HOLD;      // 100ms 이상 이어진 소리만 인정
    bool newSpike = spikeNow && !inSpike;       // 하나의 소리를 한 번만 셈
    heldRun = spikeNow ? heldRun + 1 : 0;
    if (spikeNow && heldRun % 20 == 0) newSpike = true;  // 1초 넘게 이어지면 1초마다 1회 추가
    inSpike = spikeNow;
    // 배경(백색소음) 수준은 천천히 따라감: 급증 중에는 더 느리게
    baseline += (spikeNow ? 0.001f : 0.005f) * (p2p - baseline);

    portENTER_CRITICAL(&sndMux);
    sndSum += p2p; sndSq += p2p * p2p; sndN++;
    if (p2p > sndMax) sndMax = p2p;
    secSum += p2p;
    if (++secN >= 20) {                                  // 1초(50ms × 20)마다 평균 저장
      secLvl[secHead] = secSum / secN; secHead = (secHead + 1) % SEC_RING;
      if (secFilled < SEC_RING) secFilled++;
      secSum = 0; secN = 0;
    }
    if (specOk) { double w = (double)p2p * p2p; specW += w; specSp += w * sp; specFl += w * fl; specCt += w * ct; }
    if (newSpike) { sndSpikes++; spikeT[spikeHead] = millis(); spikeHead = (spikeHead + 1) % SPK_RING; }
    portEXIT_CRITICAL(&sndMux);
    vTaskDelay(1);
  }
}

// ───────── MH-Z19C (UART 직접 명령) ─────────
uint8_t mhzChecksum(const uint8_t* p) {
  uint8_t s = 0;
  for (int i = 1; i < 8; i++) s += p[i];
  return 0xFF - s + 1;
}

int readCO2() {
  uint8_t cmd[9] = {0xFF, 0x01, 0x86, 0, 0, 0, 0, 0, 0};
  cmd[8] = mhzChecksum(cmd);
  while (co2Serial.available()) co2Serial.read();
  co2Serial.write(cmd, 9);
  uint8_t r[9];
  if (co2Serial.readBytes(r, 9) != 9) return -1;
  if (r[0] != 0xFF || r[1] != 0x86 || mhzChecksum(r) != r[8]) return -1;
  return r[2] * 256 + r[3];
}

// ───────── MH-Z19C PWM (인터럽트로 HIGH·LOW 길이 측정) ─────────
// 주기 약 1004ms. CO2 = 범위 × (Th − 2ms) / (Th + Tl − 4ms)
#ifndef CO2_PWM_RANGE
#define CO2_PWM_RANGE 5000      // 센서 측정 범위(ppm). UART 값의 약 0.4배로 나오면 2000으로 바꾸기
#endif
volatile uint32_t pwmRise = 0, pwmFall = 0, pwmTh = 0, pwmTl = 0, pwmStamp = 0;
void IRAM_ATTR co2PwmISR() {
  uint32_t now = micros();
  if (digitalRead(PIN_CO2_PWM)) {             // 올라가는 순간: 직전 LOW 길이 확정
    if (pwmFall) pwmTl = now - pwmFall;
    pwmRise = now;
  } else if (pwmRise) {                        // 내려가는 순간: HIGH 길이 확정
    pwmTh = now - pwmRise;
    pwmFall = now;
    pwmStamp = millis();
  }
}
int readCO2pwm() {
  noInterrupts();
  uint32_t th = pwmTh, tl = pwmTl, st = pwmStamp;
  interrupts();
  if (!th || !tl || millis() - st > 3000) return -1;      // 3초 넘게 신호 없음
  float thm = th / 1000.0f, tlm = tl / 1000.0f, period = thm + tlm;
  if (period < 900 || period > 1100) return -1;           // 주기가 이상하면 버림
  return (int)(CO2_PWM_RANGE * (thm - 2) / (period - 4));
}

void setCO2ABC(bool on) {
  uint8_t cmd[9] = {0xFF, 0x01, 0x79, (uint8_t)(on ? 0xA0 : 0x00), 0, 0, 0, 0, 0};
  cmd[8] = mhzChecksum(cmd);
  co2Serial.write(cmd, 9);
}

// ───────── 1초 측정 누적 ─────────
float co2Acc = 0, co2pAcc = 0, tAcc = 0, hAcc = 0, luxAcc = 0;
int   co2Cnt = 0, co2pCnt = 0, thCnt = 0, luxCnt = 0;

void readEnvOnce() {
  int c = readCO2();
  if (c > 0 && c < 10000) { co2Acc += c; co2Cnt++; }
  int cp = readCO2pwm();
  if (cp > 0 && cp < 10000) { co2pAcc += cp; co2pCnt++; }
  if (shtOk) {
#if SHT_MODEL == 40
    sensors_event_t hum, temp;
    if (sht.getEvent(&hum, &temp)) { tAcc += temp.temperature; hAcc += hum.relative_humidity; thCnt++; }
#else
    float t = sht.readTemperature(), h = sht.readHumidity();
    if (!isnan(t) && !isnan(h)) { tAcc += t; hAcc += h; thCnt++; }
#endif
  }
  if (luxOk) {
    float l = lightMeter.readLightLevel();
    if (l >= 0) { luxAcc += l; luxCnt++; }
  }
}

// ───────── 특징 계산 ─────────
const Rec& recAgo(int k) {            // k=0 이 가장 최근
  int idx = (bufHead - 1 - k + BUF_N) % BUF_N;
  return buf[idx];
}

// 최근 n개 기록의 CO2 기울기(ppm/분), 최소제곱법
float co2SlopeLast(int n) {
  n = min(n, bufCount);
  if (n < 3) return 0;
  double sx = 0, sy = 0, sxx = 0, sxy = 0; int m = 0;
  time_t t0 = recAgo(n - 1).ts;
  for (int k = n - 1; k >= 0; k--) {
    const Rec& r = recAgo(k);
    if (isnan(r.co2)) continue;
    double x = (r.ts - t0) / 60.0, y = r.co2;
    sx += x; sy += y; sxx += x * x; sxy += x * y; m++;
  }
  double d = m * sxx - sx * sx;
  if (m < 3 || fabs(d) < 1e-9) return 0;
  return (m * sxy - sx * sy) / d;
}

Features computeFeatures() {
  Features f = {0, 0, 0, 0, 0, 0, 0, 0};
  int nC = 0, nT = 0, nL = 0;
  for (int k = 0; k < bufCount; k++) {
    const Rec& r = recAgo(k);
    f.sndMean += r.sndMean; f.sndStd += r.sndStd; f.sndSpikes += r.sndSpikes;
    if (!isnan(r.co2)) { f.co2Mean += r.co2; nC++; }
    if (!isnan(r.temp)) { f.temp += r.temp; f.hum += r.hum; nT++; }
    if (!isnan(r.lux)) { f.lux += r.lux; nL++; }
  }
  if (bufCount) { f.sndMean /= bufCount; f.sndStd /= bufCount; }
  f.co2Mean = nC ? f.co2Mean / nC : NAN;
  f.temp = nT ? f.temp / nT : NAN;
  f.hum  = nT ? f.hum / nT : NAN;
  f.lux  = nL ? f.lux / nL : NAN;
  f.co2Slope = co2SlopeLast(BUF_N);
  return f;
}

// 최근 n초 평균 소리 크기 (실시간)
float soundLevel(int n) {
  float s = 0; int c = 0;
  portENTER_CRITICAL(&sndMux);
  for (int k = 0; k < min(n, secFilled); k++) { s += secLvl[(secHead - 1 - k + SEC_RING) % SEC_RING]; c++; }
  portEXIT_CRITICAL(&sndMux);
  return c ? s / c : 0;
}

// 최근 ms 밀리초 안의 급증 횟수 (실시간)
int spikesWithin(uint32_t ms) {
  uint32_t now = millis(); int c = 0;
  portENTER_CRITICAL(&sndMux);
  for (int i = 0; i < SPK_RING; i++)
    if (spikeT[i] && now - spikeT[i] < ms) c++;
  portEXIT_CRITICAL(&sndMux);
  return c;
}

// ───────── 판단 ─────────
// LCD에 보여줄 최신 값 (10초 기록 때 갱신, 화면은 3초마다 전환)
float lcdCo2 = NAN, lcdTemp = NAN, lcdHum = NAN, lcdLux = NAN, lcdSnd = NAN;
int   lcdSpk = 0, lcdPage = 0;
bool ventAlert = false, noisy = false, winOpen = false;
int servoAngle = -1;
Servo windowServo;
State state = ST_OPTIMAL;
String causeKo = "", causeLcd = "OK";
// 소음을 뺀 나머지 판단 결과 (10초마다 갱신)
State baseState = ST_OPTIMAL;
String baseCauseKo = "", baseCauseLcd = "OK";
uint32_t lastLcd = 0;
void updateNoise();
void drawLcd();
float etaMin = -1;   // 졸음 임계값까지 예상 시간(분), -1 = 해당 없음

#if USE_MLP
int mlpPredict(const Features& f) {
  float x[MLP_IN] = {f.sndMean, f.sndStd, f.sndSpikes, f.co2Mean, f.co2Slope, f.lux, f.temp, f.hum};
  for (int i = 0; i < MLP_IN; i++) {
    if (isnan(x[i])) x[i] = MLP_MEAN[i];
    x[i] = (x[i] - MLP_MEAN[i]) / MLP_STD[i];      // z-점수 표준화
  }
  float h1[MLP_H1], h2[MLP_H2], o[MLP_OUT];
  for (int j = 0; j < MLP_H1; j++) {
    float s = MLP_B1[j];
    for (int i = 0; i < MLP_IN; i++) s += MLP_W1[j][i] * x[i];
    h1[j] = s > 0 ? s : 0;                          // ReLU
  }
  for (int j = 0; j < MLP_H2; j++) {
    float s = MLP_B2[j];
    for (int i = 0; i < MLP_H1; i++) s += MLP_W2[j][i] * h1[i];
    h2[j] = s > 0 ? s : 0;
  }
  int best = 0;
  for (int j = 0; j < MLP_OUT; j++) {
    float s = MLP_B3[j];
    for (int i = 0; i < MLP_H2; i++) s += MLP_W3[j][i] * h2[i];
    o[j] = s;
    if (o[j] > o[best]) best = j;
  }
  return best;
}
#endif

void decide(const Features& f, float co2Now) {
  // 1) 환기 알림: 예측 + 히스테리시스
  float slope5 = co2SlopeLast(30);                   // 최근 5분 기울기
  etaMin = -1;
  if (!isnan(co2Now) && slope5 > 0.5 && co2Now < ctl.co2Drowsy)
    etaMin = (ctl.co2Drowsy - co2Now) / slope5;
  if (!isnan(co2Now)) {
    if (!ventAlert && (co2Now >= ctl.co2Drowsy || (etaMin >= 0 && etaMin <= ctl.etaMin))) ventAlert = true;
    else if (ventAlert && co2Now <= ctl.co2Clear) ventAlert = false;
  }
  // 3) 원인 (규칙 기반, 우선순위 순)
  bool luxLow  = !isnan(lcdLux) && lcdLux < ctl.luxLow;   // 최근 10초 조도 기준 (금방 돌아오게)
  bool tHigh   = !isnan(f.temp) && f.temp > ctl.tMax;
  bool tLow    = !isnan(f.temp) && f.temp < ctl.tMin;
  bool hBad    = !isnan(f.hum)  && (f.hum < ctl.hMin || f.hum > ctl.hMax);

  State ruleState = ST_OPTIMAL;   // 소음은 updateNoise()가 0.3초마다 따로 판단
  if (ventAlert)        { ruleState = ST_DROWSY;     baseCauseKo = "CO2 상승 - 환기";   baseCauseLcd = "CO2 - VENTILATE"; }
  else if (luxLow)      { ruleState = ST_DROWSY;     baseCauseKo = "조명 부족";         baseCauseLcd = "LIGHT LOW"; }
  else if (tHigh)       { ruleState = ST_DROWSY;     baseCauseKo = "온도 높음";         baseCauseLcd = "TEMP HIGH"; }
  else if (tLow)        { ruleState = ST_DISTRACTED; baseCauseKo = "온도 낮음";         baseCauseLcd = "TEMP LOW"; }
  else if (hBad)        { ruleState = ST_DISTRACTED; baseCauseKo = "습도 부적절";       baseCauseLcd = "HUMIDITY"; }
  else                  { baseCauseKo = "";                                            baseCauseLcd = "OK"; }

#if USE_MLP
  baseState = (State)mlpPredict(f);   // 상태는 인공신경망, 원인 설명은 규칙
#else
  baseState = ruleState;
#endif

  // 4) 창문(서보): 자동이면 환기 알림을 따라감
  if (ctl.winMode == "on") winOpen = true;
  else if (ctl.winMode == "off") winOpen = false;
  else winOpen = ventAlert;
}


void lcdLine(int row, const char* s) {
  char b[17]; snprintf(b, sizeof(b), "%-16s", s);
  lcd.setCursor(0, row); lcd.print(b);
}

void drawLcd() {
  char l1[17], l2[17];
  uint32_t up = millis() / 1000;
  if (lcdPage == 0) {
    // 1화면: CO2 · 온도 · 습도 / 원인 또는 환기 예상
    if (up < WARMUP_S) snprintf(l1, sizeof(l1), "CO2 warm %lu:%02lu", (unsigned long)(WARMUP_S - up) / 60, (unsigned long)(WARMUP_S - up) % 60);
    else snprintf(l1, sizeof(l1), "C%-4d %4.1fC %2d%%", isnan(lcdCo2) ? 0 : (int)lcdCo2,
                  isnan(lcdTemp) ? 0.0 : lcdTemp, isnan(lcdHum) ? 0 : (int)lcdHum);
    if (causeLcd == "OK" && etaMin >= 0 && etaMin < 60) snprintf(l2, sizeof(l2), "OK  vent in %2dm", (int)etaMin);
    else snprintf(l2, sizeof(l2), "%s", causeLcd.c_str());
  } else {
    // 2화면: 조도 · 소리 / 상태와 1분 급증 횟수
    snprintf(l1, sizeof(l1), "L%4dlx  S%4d", isnan(lcdLux) ? 0 : (int)lcdLux, (int)soundLevel(5));   // S = 최근 5초 평균 크기
    const char* st[] = {"OPTIMAL", "DROWSY", "DISTRACT"};
    snprintf(l2, sizeof(l2), "%-8s %2d/30s", st[state], lcdSpk);
  }
  lcdLine(0, l1); lcdLine(1, l2);
  lcdPage = 1 - lcdPage;
}

void applyOutputs(float co2Now, const Features& f) {
  int target = winOpen ? SERVO_OPEN : SERVO_CLOSED;
  if (target != servoAngle) { windowServo.write(target); servoAngle = target; }  // 바뀔 때만 움직임
  const Rec& r = recAgo(0);
  lcdCo2 = co2Now; lcdTemp = r.temp; lcdHum = r.hum; lcdLux = r.lux; lcdSnd = r.sndMean;
  updateNoise();
}

// ───────── 실시간 소음 판단 (0.3초마다) ─────────
// 켜짐: ① 최근 30초 급증이 기준(noise30) 이상  또는  ② 최근 5초 평균 크기가 loudLvl 이상(계속 시끄러움)
// 꺼짐: 최근 10초 급증 0회  그리고  5초 평균 크기가 loudLvl의 80% 미만
bool loudNow = false, statePush = false;
void updateNoise() {
  int sp30 = spikesWithin(30000), sp10 = spikesWithin(10000);
  float lvl = soundLevel(5);
  if (!loudNow && lvl >= ctl.loudLvl) loudNow = true;
  else if (loudNow && lvl < ctl.loudLvl * 0.8f) loudNow = false;
  if (!noisy && (sp30 >= ctl.noise30 || loudNow)) noisy = true;
  else if (noisy && sp10 == 0 && !loudNow) {                    // 10초 조용 → 해제하고 새로 셈
    noisy = false;
    portENTER_CRITICAL(&sndMux);
    for (int i = 0; i < SPK_RING; i++) spikeT[i] = 0;
    portEXIT_CRITICAL(&sndMux);
    sp30 = 0;
  }
  State prev = state;
  if (noisy && loudNow) { state = ST_DISTRACTED; causeKo = "계속 시끄러움";    causeLcd = "LOUD - QUIET"; }
  else if (noisy)  { state = ST_DISTRACTED; causeKo = "말소리·소음 많음"; causeLcd = "NOISY - QUIET"; }
  else       { state = baseState;     causeKo = baseCauseKo;       causeLcd = baseCauseLcd; }
  digitalWrite(PIN_LED_G, state == ST_OPTIMAL);
  digitalWrite(PIN_LED_Y, state == ST_DROWSY);
  digitalWrite(PIN_LED_R, state == ST_DISTRACTED);
  lcdSpk = sp30;
  if (state != prev) { lcdPage = 0; drawLcd(); lastLcd = millis(); statePush = true; }   // 바뀌면 화면·웹 바로
}

// ───────── Firebase (REST) ─────────
int fbRequest(const char* method, const String& path, const String& body, String* resp) {
  if (WiFi.status() != WL_CONNECTED) return -1;
  WiFiClientSecure client;
  client.setInsecure();               // 학교 프로젝트용: 인증서 검증 생략
  HTTPClient http;
  String url = String(FIREBASE_URL) + path + ".json?auth=" + FIREBASE_SECRET;
  if (!http.begin(client, url)) return -1;
  http.setTimeout(6000);
  http.addHeader("Content-Type", "application/json");
  int code = http.sendRequest(method, body);
  if (resp && code == 200) *resp = http.getString();
  http.end();
  return code;
}

String pending[PENDING_MAX];
int pendingCount = 0;

void queueRecord(const String& kv) {     // kv = "\"logs/1731..\":{...}"
  if (pendingCount == PENDING_MAX) {     // 가장 오래된 것 버림
    for (int i = 1; i < PENDING_MAX; i++) pending[i - 1] = pending[i];
    pendingCount--;
  }
  pending[pendingCount++] = kv;
}

void uploadAll(const String& liveJson) {
  String body = "{\"live\":" + liveJson;
  for (int i = 0; i < pendingCount; i++) body += "," + pending[i];
  body += "}";
  int code = fbRequest("PATCH", "/", body, nullptr);  // 루트("/")에 여러 위치를 한 번에 갱신
  if (code == 200) pendingCount = 0;
  else Serial.printf("[업로드 실패] code=%d, 보관 중 %d개\n", code, pendingCount);
}

// 상태가 바뀌면 10초 기록을 기다리지 않고 웹에 상태만 바로 올림
void pushState() {
  String body = "{\"state_ko\":\"" + String(STATE_KO[state]) + "\",\"cause\":\"" + causeKo +
                "\",\"rec/state\":" + String((int)state) + "}";
  fbRequest("PATCH", "/live", body, nullptr);
}

void fetchControl() {
  String body;
  if (fbRequest("GET", "/control", "", &body) != 200 || body == "null") return;
  JsonDocument doc;
  if (deserializeJson(doc, body)) return;
  ctl.co2Drowsy = doc["co2Drowsy"] | ctl.co2Drowsy;
  ctl.co2Clear  = doc["co2Clear"]  | ctl.co2Clear;
  ctl.etaMin    = doc["etaMin"]    | ctl.etaMin;
  ctl.luxLow    = doc["luxLow"]    | ctl.luxLow;
  ctl.tMin      = doc["tMin"]      | ctl.tMin;
  ctl.tMax      = doc["tMax"]      | ctl.tMax;
  ctl.hMin      = doc["hMin"]      | ctl.hMin;
  ctl.hMax      = doc["hMax"]      | ctl.hMax;
  ctl.noise30   = doc["noise30"]   | ctl.noise30;
  ctl.loudLvl   = doc["loudLvl"]   | ctl.loudLvl;
  const char* wm = doc["winMode"] | "auto";
  ctl.winMode = wm;
}

String jnum(float v, int d = 1) {        // NaN 은 null 로
  if (isnan(v)) return "null";
  return String(v, d);
}

// ───────── 10초 기록 ─────────
void makeRecord() {
  Rec r;
  r.ts = time(nullptr);

  portENTER_CRITICAL(&sndMux);
  float s = sndSum, sq = sndSq, mx = sndMax; int n = sndN, spk = sndSpikes;
  sndSum = sndSq = sndMax = 0; sndN = sndSpikes = 0;
  double sw = specW, ssp = specSp, sfl = specFl, sct = specCt;
  specW = specSp = specFl = specCt = 0;
  portEXIT_CRITICAL(&sndMux);
  r.sndSp   = sw > 0 ? ssp / sw : NAN;
  r.sndFlat = sw > 0 ? sfl / sw : NAN;
  r.sndCent = sw > 0 ? sct / sw : NAN;
  r.sndMean = n ? s / n : NAN;
  r.sndStd  = n ? sqrtf(fmaxf(sq / n - r.sndMean * r.sndMean, 0)) : NAN;
  r.sndMax  = mx;
  r.sndSpikes = spk;

  r.co2  = co2Cnt ? co2Acc / co2Cnt : NAN;
  r.co2pwm = co2pCnt ? co2pAcc / co2pCnt : NAN;
  r.temp = thCnt ? tAcc / thCnt : NAN;
  r.hum  = thCnt ? hAcc / thCnt : NAN;
  r.lux  = luxCnt ? luxAcc / luxCnt : NAN;
  co2Acc = co2pAcc = tAcc = hAcc = luxAcc = 0; co2Cnt = co2pCnt = thCnt = luxCnt = 0;

  uint32_t up = millis() / 1000;
  bool warm = up < WARMUP_S;
  if (warm) { r.co2 = NAN; r.co2pwm = NAN; }   // 예열 중 CO2 는 판단에 쓰지 않음

  buf[bufHead] = r;
  bufHead = (bufHead + 1) % BUF_N;
  if (bufCount < BUF_N) bufCount++;

  lcdLux = r.lux;
  Features f = computeFeatures();
  decide(f, r.co2);
  applyOutputs(r.co2, f);

  // 리드 스위치: 1 = 열림(자석이 떨어짐 → 풀업으로 HIGH)
  String reed = "";
  for (int i = 0; i < REED_COUNT; i++) reed += digitalRead(REED_PINS[i]) ? '1' : '0';

  String rec = "{\"ts\":" + String((long)r.ts) +
    ",\"up\":" + String(up) +
    ",\"flag\":\"" + (warm ? "warmup" : "ok") + "\"" +
    ",\"co2\":" + jnum(r.co2, 0) +
    ",\"co2_pwm\":" + jnum(r.co2pwm, 0) +
    ",\"temp\":" + jnum(r.temp, 2) +
    ",\"hum\":" + jnum(r.hum, 2) +
    ",\"lux\":" + jnum(r.lux, 1) +
    ",\"snd_mean\":" + jnum(r.sndMean, 1) +
    ",\"snd_std\":" + jnum(r.sndStd, 1) +
    ",\"snd_max\":" + jnum(r.sndMax, 0) +
    ",\"snd_spikes\":" + String(r.sndSpikes) +
    ",\"snd_sp\":" + jnum(r.sndSp, 3) +
    ",\"snd_flat\":" + jnum(r.sndFlat, 3) +
    ",\"snd_cent\":" + jnum(r.sndCent, 0) +
    ",\"reed\":\"" + reed + "\"" +
    ",\"state\":" + String((int)state) +
    ",\"vent\":" + (ventAlert ? "true" : "false") +
    ",\"win\":" + (winOpen ? "true" : "false") + "}";

  String live = "{\"rec\":" + rec +
    ",\"state_ko\":\"" + STATE_KO[state] + "\"" +
    ",\"cause\":\"" + causeKo + "\"" +
    ",\"eta_min\":" + jnum(etaMin, 1) +
    ",\"model\":\"" + (USE_MLP ? "mlp" : "rule") + "\"" +
    ",\"feat\":{\"snd_mean\":" + jnum(f.sndMean) + ",\"snd_std\":" + jnum(f.sndStd) +
      ",\"snd_spikes\":" + jnum(f.sndSpikes, 0) + ",\"co2_mean\":" + jnum(f.co2Mean, 0) +
      ",\"co2_slope\":" + jnum(f.co2Slope, 2) + ",\"lux\":" + jnum(f.lux) +
      ",\"temp\":" + jnum(f.temp, 2) + ",\"hum\":" + jnum(f.hum, 2) + "}}";

  Serial.println(rec);
  if (r.ts > 1700000000) {               // 시계(NTP)가 맞춰진 뒤에만 저장
    queueRecord("\"logs/" + String((long)r.ts) + "\":" + rec);
    uploadAll(live);
  }
}

// ───────── 와이파이 ─────────
// ───────── 와이파이: 여러 개 자동 연결 + 장치 웹페이지에서 추가 ─────────
// · 기본 와이파이(config의 WIFI_LIST) + 웹앱에서 추가한 와이파이 중 신호가 가장 센 곳에 자동 연결
// · 웹앱 '와이파이 관리'에서 추가 → Firebase /wifi (쓰기 키 없이는 읽을 수 없음)
//   → 장치가 연결돼 있을 때 1분마다 받아서 내부(Preferences)에 저장 → 다음에 그곳에 가면 자동 연결
// · 비상용: 끊긴 지 1분이 지나면 설정용 와이파이 'ENV-<BOARD_NAME>' 을 켬 → 접속하면 설정 페이지
#ifndef WIFI_LIST
#define WIFI_LIST {WIFI_SSID, WIFI_PASS}
#endif
#ifndef BOARD_NAME
#define BOARD_NAME "main"
#endif
#ifndef PORTAL_KEY
#define PORTAL_KEY "classenv7"   // 관리 비밀번호 겸 설정용 와이파이 비밀번호(8자 이상)
#endif
struct WifiCred { const char* ssid; const char* pass; };
const WifiCred BUILTIN_WIFI[] = { WIFI_LIST };
const int BUILTIN_COUNT = sizeof(BUILTIN_WIFI) / sizeof(BUILTIN_WIFI[0]);
const int MAX_SAVED = 8;
const uint32_t PORTAL_AFTER_MS = 60000;   // 끊긴 지 1분 → 설정 모드
const uint32_t RETRY_EVERY_MS  = 30000;   // 끊겨 있으면 30초마다 재연결

Preferences wifiPrefs;
WiFiMulti*  wifiMulti = nullptr;
WebServer   webServer(80);
DNSServer   dnsServer;
String savedSsid[MAX_SAVED], savedPass[MAX_SAVED];
int savedCount = 0;
bool portalOn = false, tryNow = false, wasConnected = false;
uint32_t lostSince = 0, lastRetry = 0;
String apName = String("ENV-") + BOARD_NAME;

void loadSavedWifi() {
  wifiPrefs.begin("wifi", true);
  savedCount = min((int)wifiPrefs.getInt("n", 0), MAX_SAVED);
  for (int i = 0; i < savedCount; i++) {
    savedSsid[i] = wifiPrefs.getString(("s" + String(i)).c_str(), "");
    savedPass[i] = wifiPrefs.getString(("p" + String(i)).c_str(), "");
  }
  wifiPrefs.end();
}
void writeSavedWifi() {
  wifiPrefs.begin("wifi", false);
  wifiPrefs.clear();
  wifiPrefs.putInt("n", savedCount);
  for (int i = 0; i < savedCount; i++) {
    wifiPrefs.putString(("s" + String(i)).c_str(), savedSsid[i]);
    wifiPrefs.putString(("p" + String(i)).c_str(), savedPass[i]);
  }
  wifiPrefs.end();
}
void rebuildWifiList() {
  delete wifiMulti;
  wifiMulti = new WiFiMulti();
  for (int i = 0; i < BUILTIN_COUNT; i++) wifiMulti->addAP(BUILTIN_WIFI[i].ssid, BUILTIN_WIFI[i].pass);
  for (int i = 0; i < savedCount; i++)    wifiMulti->addAP(savedSsid[i].c_str(), savedPass[i].c_str());
}
void startPortal() {
  if (portalOn) return;
  WiFi.mode(WIFI_AP_STA);
  WiFi.softAP(apName.c_str(), PORTAL_KEY);
  dnsServer.start(53, "*", WiFi.softAPIP());
  portalOn = true;
  Serial.printf("[WiFi] 설정 모드: 와이파이 '%s' (비번 %s) 접속 → http://%s\n",
                apName.c_str(), PORTAL_KEY, WiFi.softAPIP().toString().c_str());
}
void stopPortal() {
  if (!portalOn) return;
  dnsServer.stop();
  WiFi.softAPdisconnect(true);
  WiFi.mode(WIFI_STA);
  portalOn = false;
  Serial.println("[WiFi] 설정 모드 종료");
}
bool tryConnect(uint32_t timeoutMs) {
  Serial.println("[WiFi] 연결 시도...");
  if (wifiMulti->run(timeoutMs) == WL_CONNECTED) {
    Serial.printf("[WiFi] 연결됨: %s  IP: %s  (%d dBm)\n",
                  WiFi.SSID().c_str(), WiFi.localIP().toString().c_str(), WiFi.RSSI());
    return true;
  }
  Serial.println("[WiFi] 연결 실패");
  return false;
}

String htmlEsc(const String& s) {
  String o;
  for (char c : s) {
    if (c == '<') o += "&lt;"; else if (c == '>') o += "&gt;";
    else if (c == '&') o += "&amp;"; else if (c == '"') o += "&quot;";
    else o += c;
  }
  return o;
}
String pageHead(const String& title) {
  return "<!DOCTYPE html><html><head><meta charset='utf-8'>"
         "<meta name='viewport' content='width=device-width,initial-scale=1'>"
         "<title>" + title + "</title><style>"
         "body{font-family:sans-serif;max-width:480px;margin:16px auto;padding:0 12px;color:#222}"
         ".box{background:#f2f4f7;border-radius:8px;padding:10px 12px;margin:10px 0}"
         "input{width:100%;padding:9px;margin:4px 0 10px;box-sizing:border-box;font-size:16px}"
         "button{padding:9px 14px;font-size:15px;border:0;border-radius:6px;background:#2b6cd8;color:#fff}"
         ".del{background:#d9534f;padding:5px 10px;font-size:13px}"
         "li{margin:6px 0;display:flex;justify-content:space-between;align-items:center}"
         "ul{padding-left:0;list-style:none}.muted{color:#888;font-size:13px}"
         "</style></head><body>";
}
void handleRoot() {
  String h = pageHead("학습환경조절기 WiFi");
  h += "<h2>학습 환경 조절기 '" + htmlEsc(BOARD_NAME) + "' 와이파이</h2><div class='box'>";
  if (WiFi.status() == WL_CONNECTED)
    h += "✅ 연결됨: <b>" + htmlEsc(WiFi.SSID()) + "</b><br>IP " + WiFi.localIP().toString() +
         " · 신호 " + String(WiFi.RSSI()) + " dBm";
  else h += "❌ 연결 안 됨";
  if (portalOn) h += "<br><span class='muted'>설정 모드 켜짐</span>";
  h += "</div><h3>기본 와이파이</h3><ul>";
  for (int i = 0; i < BUILTIN_COUNT; i++) h += "<li>" + htmlEsc(BUILTIN_WIFI[i].ssid) + "</li>";
  h += "</ul><h3>추가한 와이파이 (" + String(savedCount) + "/" + String(MAX_SAVED) + ")</h3><ul>";
  if (savedCount == 0) h += "<li class='muted'>없음</li>";
  for (int i = 0; i < savedCount; i++) {
    h += "<li>" + htmlEsc(savedSsid[i]) +
         "<form method='POST' action='/del' style='margin:0' onsubmit=\"var k=prompt('관리 비밀번호');"
         "if(!k)return false;this.key.value=k;\">"
         "<input type='hidden' name='i' value='" + String(i) + "'>"
         "<input type='hidden' name='key'><button class='del'>삭제</button></form></li>";
  }
  h += "</ul>";
  int n = WiFi.scanNetworks();
  h += "<h3>와이파이 추가</h3><form method='POST' action='/add'>"
       "이름 (SSID)<input name='ssid' list='near' required maxlength='32' autocomplete='off'><datalist id='near'>";
  for (int i = 0; i < n; i++) h += "<option value=\"" + htmlEsc(WiFi.SSID(i)) + "\">";
  WiFi.scanDelete();
  h += "</datalist>비밀번호 (없으면 비워 두기)<input name='pass' type='password' maxlength='63'>"
       "관리 비밀번호<input name='key' type='password' required>"
       "<button>저장하고 연결</button></form>"
       "<p class='muted'>같은 이름을 다시 저장하면 비밀번호가 바뀝니다. 신호가 가장 센 곳에 자동 연결됩니다.</p>"
       "</body></html>";
  webServer.send(200, "text/html; charset=utf-8", h);
}
void sendMsg(const String& msg) {
  webServer.send(200, "text/html; charset=utf-8",
    pageHead("결과") + "<div class='box'>" + msg + "</div><p><a href='/'>← 돌아가기</a></p></body></html>");
}
void handleAdd() {
  if (webServer.arg("key") != PORTAL_KEY) { sendMsg("관리 비밀번호가 틀렸어요."); return; }
  String ssid = webServer.arg("ssid"); ssid.trim();
  String pass = webServer.arg("pass");
  if (ssid.length() == 0 || ssid.length() > 32) { sendMsg("와이파이 이름이 비었거나 너무 길어요."); return; }
  if (pass.length() > 0 && (pass.length() < 8 || pass.length() > 63)) { sendMsg("비밀번호는 8~63자여야 해요 (없으면 비워 두기)."); return; }
  for (int i = 0; i < BUILTIN_COUNT; i++)
    if (ssid == BUILTIN_WIFI[i].ssid) { sendMsg("이미 기본으로 들어 있는 와이파이예요."); return; }
  int idx = -1;
  for (int i = 0; i < savedCount; i++) if (savedSsid[i] == ssid) idx = i;
  if (idx < 0) {
    if (savedCount >= MAX_SAVED) { sendMsg("최대 개수가 찼어요. 하나 삭제하고 다시 해 주세요."); return; }
    idx = savedCount++;
  }
  savedSsid[idx] = ssid; savedPass[idx] = pass;
  writeSavedWifi(); rebuildWifiList();
  if (WiFi.status() != WL_CONNECTED) tryNow = true;
  Serial.printf("[WiFi] 추가/수정: %s\n", ssid.c_str());
  sendMsg("저장했어요: <b>" + htmlEsc(ssid) + "</b><br>연결이 안 돼 있으면 바로 연결을 시도해요.");
}
void handleDel() {
  if (webServer.arg("key") != PORTAL_KEY) { sendMsg("관리 비밀번호가 틀렸어요."); return; }
  int i = webServer.arg("i").toInt();
  if (i < 0 || i >= savedCount) { sendMsg("잘못된 항목이에요."); return; }
  String name = savedSsid[i];
  for (int k = i; k < savedCount - 1; k++) { savedSsid[k] = savedSsid[k + 1]; savedPass[k] = savedPass[k + 1]; }
  savedCount--;
  writeSavedWifi(); rebuildWifiList();
  sendMsg("삭제했어요: <b>" + htmlEsc(name) + "</b>");
}
void handleNotFound() {
  if (portalOn) {
    webServer.sendHeader("Location", "http://" + WiFi.softAPIP().toString() + "/", true);
    webServer.send(302, "text/plain", "");
  } else webServer.send(404, "text/plain", "not found");
}
// 웹앱에서 관리하는 와이파이 목록(/wifi)을 받아 장치에 저장
int fbRequest(const char* method, const String& path, const String& body, String* resp);
void syncWifiFromCloud() {
  String body;
  if (fbRequest("GET", "/wifi", "", &body) != 200) return;
  String ns[MAX_SAVED], np[MAX_SAVED]; int n = 0;
  if (body != "null") {
    JsonDocument doc;
    if (deserializeJson(doc, body)) return;
    for (JsonPair kv : doc.as<JsonObject>()) {
      if (n >= MAX_SAVED) break;
      const char* ss = kv.value()["ssid"] | "";
      const char* pp = kv.value()["pass"] | "";
      if (!*ss) continue;
      ns[n] = ss; np[n] = pp; n++;
    }
  }
  bool same = (n == savedCount);
  for (int i = 0; same && i < n; i++) same = (ns[i] == savedSsid[i] && np[i] == savedPass[i]);
  if (same) return;
  for (int i = 0; i < n; i++) { savedSsid[i] = ns[i]; savedPass[i] = np[i]; }
  savedCount = n;
  writeSavedWifi(); rebuildWifiList();
  Serial.printf("[WiFi] 웹앱 와이파이 목록 반영: %d개\n", n);
}

void wifiBegin() {
  WiFi.mode(WIFI_STA);
  loadSavedWifi();
  rebuildWifiList();
  Serial.printf("[WiFi] 등록된 와이파이: 기본 %d개 + 추가 %d개\n", BUILTIN_COUNT, savedCount);
  if (!tryConnect(20000)) {
    lostSince = millis() - PORTAL_AFTER_MS;
    startPortal();
  }
  webServer.on("/", HTTP_GET, handleRoot);
  webServer.on("/add", HTTP_POST, handleAdd);
  webServer.on("/del", HTTP_POST, handleDel);
  webServer.onNotFound(handleNotFound);
  webServer.begin();
}
void wifiLoop() {
  webServer.handleClient();
  if (portalOn) dnsServer.processNextRequest();
  if (WiFi.status() == WL_CONNECTED) {
    if (!wasConnected) { wasConnected = true; Serial.printf("[WiFi] 연결됨: %s  IP: %s\n", WiFi.SSID().c_str(), WiFi.localIP().toString().c_str()); }
    lostSince = 0;
    if (portalOn && WiFi.softAPgetStationNum() == 0) stopPortal();
    return;
  }
  wasConnected = false;
  if (lostSince == 0) { lostSince = millis(); Serial.println("[WiFi] 연결 끊김"); }
  if (!portalOn && millis() - lostSince > PORTAL_AFTER_MS) startPortal();
  bool someoneOnPortal = portalOn && WiFi.softAPgetStationNum() > 0;
  if (tryNow || (!someoneOnPortal && millis() - lastRetry > RETRY_EVERY_MS)) {
    tryNow = false; lastRetry = millis();
    tryConnect(8000);
  }
}

// ───────── setup / loop ─────────
void setup() {
  Serial.begin(115200);
  pinMode(PIN_LED_R, OUTPUT); pinMode(PIN_LED_Y, OUTPUT); pinMode(PIN_LED_G, OUTPUT);
  windowServo.setPeriodHertz(50);
  windowServo.attach(PIN_SERVO, 500, 2400);
  windowServo.write(SERVO_CLOSED); servoAngle = SERVO_CLOSED;
  for (int i = 0; i < REED_COUNT; i++) pinMode(REED_PINS[i], INPUT_PULLUP);

  analogReadResolution(12);
  analogSetPinAttenuation(PIN_MIC, ADC_11db);     // 0~3.3V 범위

  Wire.begin(21, 22);
  lcd.init(); lcd.backlight();
  lcd.setCursor(0, 0); lcd.print("Classroom Env");
  lcd.setCursor(0, 1); lcd.print("starting...");

#if SHT_MODEL == 40
  shtOk = sht.begin();
#else
  shtOk = sht.begin(0x44) || sht.begin(0x45);
#endif
  luxOk = lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE);
  Serial.printf("SHT:%s  BH1750:%s\n", shtOk ? "OK" : "없음", luxOk ? "OK" : "없음");

  co2Serial.begin(9600, SERIAL_8N1, PIN_CO2_RX, PIN_CO2_TX);
  co2Serial.setTimeout(200);
#if CO2_ABC_OFF
  delay(100); setCO2ABC(false);
#endif
  pinMode(PIN_CO2_PWM, INPUT);
  attachInterrupt(digitalPinToInterrupt(PIN_CO2_PWM), co2PwmISR, CHANGE);

  wifiBegin();
  configTzTime("KST-9", "pool.ntp.org", "time.google.com");

  // 신호등 점검: 빨-노-초 순서로 한 번씩
  int leds[3] = {PIN_LED_R, PIN_LED_Y, PIN_LED_G};
  for (int i = 0; i < 3; i++) { digitalWrite(leds[i], HIGH); delay(300); digitalWrite(leds[i], LOW); }

  xTaskCreatePinnedToCore(soundTask, "sound", 4096, nullptr, 1, nullptr, 1);
}

uint32_t lastEnv = 0, lastRec = 0, lastCtl = 0, lastNoise = 0, lastPush = 0, lastWifiSync = 0;

void loop() {
  uint32_t now = millis();
  wifiLoop();
  if (now - lastEnv >= ENV_READ_MS) { lastEnv = now; readEnvOnce(); }
  if (now - lastRec >= RECORD_MS)   { lastRec = now; makeRecord(); }
  if (now - lastCtl >= CONTROL_MS)  { lastCtl = now; fetchControl(); }
  if (WiFi.status() == WL_CONNECTED && (lastWifiSync == 0 || now - lastWifiSync >= 60000)) { lastWifiSync = now ? now : 1; syncWifiFromCloud(); }
  if (now - lastNoise >= 300)       { lastNoise = now; updateNoise(); }
  if (statePush && now - lastPush >= 2000) { statePush = false; lastPush = now; pushState(); }
  if (now - lastLcd >= 3000)        { lastLcd = now; drawLcd(); }
  delay(5);
}
