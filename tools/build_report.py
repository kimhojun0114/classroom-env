import markdown, re, weasyprint, os
R="/home/claude/repo"
md=open(f"{R}/docs/report.md",encoding="utf-8").read()
import re as _re
def fig(name, cap, w="100%"):
    return f'\n\n<figure><img src="file://{R}/results/{name}" style="width:{w}"><figcaption>{cap}</figcaption></figure>\n\n'
ins = [
 ("### 3-4. 프롬프트 설계 개선", fig("01_timeline.png","그림 1. 이 장치의 전체 측정 기록 (11/14~15, 회색 = 외출로 방이 빈 시간, 보라 선 = 라벨 입력)","72%")+
   fig("01_co2_uart_vs_pwm.png","그림 2. 같은 CO₂ 센서를 UART와 PWM 두 방식으로 읽은 값 비교","72%")+
   fig("04_classroom_daily_profile.png","그림 3. 7반 교실 시간대별 평균 CO₂ (5일)","88%"), "before"),
 ("**분석**\n- **기존 규칙의 오판", fig("03_confusion.png","그림 4. 혼동 행렬 — 기존 규칙은 실제로 졸리지 않았던 시간을 대부분 '졸림'으로 판단함"), "before"),
 ("### 4-3. 소리 AI", fig("mic_fft_test.png","그림 5. 소리 종류별 주파수 특징 검증 실험 — 크기(맨 위)로는 비슷한 소리가 주파수 특징으로 구분됨","66%"), "before"),
 ("**결과**\n\n| 방법 | 전체 MAE", fig("02_gd_learning_rate.png","그림 7. 학습률에 따른 경사하강 — η < 2/λmax 이면 수렴, 넘으면 발산","70%"), "before"),
 ("### 4-4. 미리 대응하기", fig("05_sound_ai_test.png","그림 6. 학습에서 뺀 소리 실험 구간의 소음 부담 — 소리 AI는 조용해지면 바로 내려감","62%"), "before"),
 ("### 4-5. 생성형 AI를 활용한 최적 조건 탐색 결과", fig("02_forecast_curve.png","그림 8. 학습에 쓰지 않은 날(11/5)의 10분 뒤 CO₂ 예측"), "before"),
 ("**결과**\n- 24시간 기록에 적용해 보니", fig("06_co2_weight.png","그림 11. CO₂ 비중에 따른 점수와 실제 상태의 일치도","58%"), "before"),
 ("### 5-2. 주요 오류와 수정", fig("06_env_score_timeline.png","그림 12. 24시간 기록에 적용한 환경 부담 점수(검은 선, CO₂ 예측 신경망 + 소리 AI + 법정 기준)와 상태 분류 신경망 점수(보라, 비교용)"), "before"),
 ("**하루 환기 일정 제안**", '<div class="two">'+fig("04_window_ventilation.png","그림 9. 열린 창문 개수와 환기 속도 (82번)")+fig("04_optimal_map.png","그림 10. 인공신경망이 본 '최적' 확률 (검은 선 = 70%)")+'</div>', "before"),
]
for key, html, pos in ins:
    i=md.index(key); md=md[:i]+html+md[i:]
md=_re.sub(r"(?m)^(?!\s*(?:[-*] |[|>]|\d+\. ))(.+)\n(?=(- |\d+\. ))", r"\1\n\n", md)   # 목록 앞에 빈 줄
md=_re.sub(r"(?m)^  - ", "    - ", md)
body=markdown.markdown(md, extensions=["tables","md_in_html"])
body=body.replace('<p>&lt;div class="two"&gt;</p>','<div class="two">')
css='''@page { size: A4; margin: 14mm 15mm 14mm; @bottom-center { content: counter(page) " / " counter(pages); font-size: 8pt; color: #888; } }
body { font-family: "Noto Sans CJK KR"; font-size: 9.6pt; line-height: 1.6; color: #1d1d1f; }
h1 { font-size: 16pt; border-bottom: 2.5px solid #c8344a; padding-bottom: 5px; margin: 0 0 6px; }
h2 { font-size: 12.5pt; color: #1f6e5a; margin: 16px 0 6px; border-left: 4px solid #1f6e5a; padding-left: 7px; }
h3 { font-size: 10.6pt; color: #a52a3c; margin: 12px 0 4px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 8px; font-size: 8.8pt; page-break-inside: avoid; }
th { background: #eef5f2; color: #1f6e5a; } th, td { border: 1px solid #cfdcd7; padding: 3px 6px; text-align: left; vertical-align: top; }
blockquote { margin: 8px 0; padding: 7px 12px; background: #fbeef0; border-left: 4px solid #c8344a; font-size: 10pt; }
code { font-size: 8.4pt; background: #f2f2f2; padding: 0 3px; border-radius: 2px; }
figure { margin: 6px 0 8px; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; } figcaption { font-size: 8.2pt; color: #555; margin-top: 2px; }
.two { display: flex; gap: 8px; } .two figure { flex: 1; margin: 6px 0; }
ul, ol { margin: 3px 0 6px; padding-left: 18px; } li { margin: 1px 0; }
hr { border: none; border-top: 1px solid #ddd; margin: 8px 0; }
p { margin: 4px 0 6px; }'''
html=f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>{css}</style></head><body>{body}</body></html>'
open("/home/claude/build/report.html","w",encoding="utf-8").write(html)
weasyprint.HTML(string=html, base_url="/").write_pdf("/home/claude/build/학습환경조절기_결과보고서_20713김호준.pdf")
print("ok")
