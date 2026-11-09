// config.example.h
// 이 파일을 복사해서 config.h 로 이름을 바꾼 뒤 값을 채우세요.
// config.h 는 .gitignore 에 들어 있어서 깃허브에 올라가지 않습니다(비밀번호 보호).
#pragma once

// ── 와이파이 ─────────────────────────────
#define WIFI_SSID   "교실와이파이이름"
#define WIFI_PASS   "와이파이비밀번호"

// ── Firebase Realtime Database ───────────
// 예: "https://classroom-env-default-rtdb.asia-southeast1.firebasedatabase.app"
// (끝에 / 붙이지 않기)
#define FIREBASE_URL    "https://classroom-env-default-rtdb.asia-southeast1.firebasedatabase.app"
// 프로젝트 설정 > 서비스 계정 > 데이터베이스 비밀번호(Database secrets)
#define FIREBASE_SECRET "YOUR_DATABASE_SECRET"

// ── 센서 선택 ────────────────────────────
// 갖고 있는 SHT 모델: 31 = SHT30/31/35 계열, 40 = SHT40/41/45 계열
#define SHT_MODEL 31

// LCD I2C 주소 (화면이 안 나오면 0x3F 로 바꿔 보기)
#define LCD_ADDR 0x27

// MH-Z19C 자동 보정(ABC) 끄기: 교실은 하루 한 번도 400ppm 바깥 공기 수준까지
// 내려가지 않을 수 있어서, 켜 두면 기준점이 틀어질 수 있음. 1 = 끄기(권장)
#define CO2_ABC_OFF 1

// 창문 리드 스위치 개수 (0~8). 데이터과학 장치를 합칠 때 사용.
// 핀 순서: 13, 14, 18, 19, 32, 33, 4, 5  (스위치 한쪽은 핀, 다른 쪽은 GND)
#define REED_COUNT 0

// 학습된 인공신경망 사용 여부. tools/export_mlp.py 로 model_weights.h 를 만든 뒤 1로.
#define USE_MLP 0
