"""분석 공통 설정: 한글 글꼴, 경로, 라벨 규칙, 특징 계산"""
import glob, os
import numpy as np, pandas as pd
import matplotlib
try:
    get_ipython  # 주피터에서는 그래프를 노트북 안에 표시
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("ENV_RAW", os.path.join(ROOT, "data/raw"))
PROC = os.environ.get("ENV_PROC", os.path.join(ROOT, "data/processed"))
RES = os.environ.get("ENV_RES", os.path.join(ROOT, "results"))
os.makedirs(PROC, exist_ok=True); os.makedirs(RES, exist_ok=True)

for f in fm.findSystemFonts():
    if "NotoSansCJK-Regular" in f or "NanumGothic" in f or "AppleSDGothicNeo" in f:
        fm.fontManager.addfont(f)
_names = {f.name for f in fm.fontManager.ttflist}
for cand in ("Noto Sans CJK JP", "NanumGothic", "Apple SD Gothic Neo", "Malgun Gothic"):
    if cand in _names:
        plt.rcParams["font.family"] = [cand, "DejaVu Sans"]; break   # 한글 글꼴에 없는 기호(₂ 등)는 DejaVu로
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

STATE_KO = {0: "최적", 1: "졸림", 2: "산만"}
STATE_COLOR = {0: "#2e9e5b", 1: "#e0a800", 2: "#d64545"}

# 장치 펌웨어와 같은 순서의 기본 특징 8개 + 주파수 특징 3개
BASE_FEATS = ["snd_mean", "snd_std", "snd_spikes", "co2_mean", "co2_slope", "lux", "temp", "hum"]
SPEC_FEATS = ["snd_sp", "snd_flat", "snd_cent"]

def label_to_state(focus, drowsy):
    """라벨 규칙: 집중도 ≥ 4 → 최적, 졸림 ≥ 4 → 졸림, 그 외 → 산만 (졸림이 우선)"""
    if drowsy >= 4: return 1
    if focus >= 4: return 0
    return 2

def read_csvs(prefix):
    files = sorted(glob.glob(os.path.join(RAW, f"{prefix}_*.csv")))
    if not files: raise FileNotFoundError(f"data/raw/{prefix}_*.csv 가 없습니다")
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)

def rolling_slope(t_min, y):
    """최소제곱 기울기 (ppm/분). t_min, y: 같은 길이 배열, NaN 제외"""
    m = ~np.isnan(y)
    if m.sum() < 3: return 0.0
    x, yy = t_min[m], y[m]
    d = len(x) * (x * x).sum() - x.sum() ** 2
    if abs(d) < 1e-9: return 0.0
    return (len(x) * (x * yy).sum() - x.sum() * yy.sum()) / d

def window_features(df, n=60):
    """장치와 똑같이 '최근 n개(10초×60=10분)' 기록으로 특징 계산. df는 시간순, 연속 구간 하나"""
    out = []
    ts = df["ts"].to_numpy(float); co2 = df["co2"].to_numpy(float)
    for i in range(len(df)):
        lo = max(0, i - n + 1); w = df.iloc[lo:i + 1]
        t = (ts[lo:i + 1] - ts[lo]) / 60.0
        out.append({
            "snd_mean": w["snd_mean"].mean(), "snd_std": w["snd_std"].mean(),
            "snd_spikes": w["snd_spikes"].sum(), "co2_mean": w["co2"].mean(),
            "co2_slope": rolling_slope(t, co2[lo:i + 1]), "lux": w["lux"].mean(),
            "temp": w["temp"].mean(), "hum": w["hum"].mean(),
            "snd_sp": w["snd_sp"].mean(), "snd_flat": w["snd_flat"].mean(), "snd_cent": w["snd_cent"].mean(),
        })
    return pd.DataFrame(out, index=df.index)
