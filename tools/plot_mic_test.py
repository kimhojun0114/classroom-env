import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from datetime import datetime, timezone, timedelta
for f in fm.findSystemFonts():
    if "NotoSansCJK-Regular" in f: fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
KST = timezone(timedelta(hours=9))
d = sorted(json.load(open("data/raw/mic_fft_test_20261114.json")).values(), key=lambda r: r["ts"])
t = [datetime.fromtimestamp(r["ts"], KST) for r in d]
segs = [("조용", "13:50:55", "13:51:24", "#9aa5b1"), ("말소리", "13:51:24", "13:51:45", "#e4572e"),
        ("쉬이이(백색소음)", "13:51:45", "13:52:07", "#7b61ff"), ("휘파람", "13:52:07", "13:52:27", "#17a398"),
        ("키보드", "13:52:27", "13:53:02", "#f3a712"), ("조용", "13:53:02", "13:53:30", "#9aa5b1")]
def ts(s): h, m, sec = map(int, s.split(":")); return datetime(2026, 11, 14, h, m, sec, tzinfo=KST)
fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
rows = [("snd_mean", "소리 크기(진폭)"), ("snd_sp", "말소리 대역 비율"), ("snd_flat", "스펙트럼 평탄도"), ("snd_cent", "스펙트럼 중심(Hz)")]
for a, (k, lab) in zip(ax, rows):
    for name, s, e, c in segs: a.axvspan(ts(s), ts(e), color=c, alpha=.15, lw=0)
    a.plot(t, [r[k] for r in d], "-o", color="#1f2933", ms=4, lw=1.6)
    a.set_ylabel(lab); a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
for name, s, e, c in segs:
    mid = ts(s) + (ts(e) - ts(s)) / 2
    ax[0].text(mid, ax[0].get_ylim()[1] * .97, name, ha="center", va="top", fontsize=9, color=c, fontweight="bold")
import matplotlib.dates as md
ax[-1].xaxis.set_major_formatter(md.DateFormatter("%H:%M:%S", tz=KST))
fig.suptitle("마이크 주파수 특징 테스트 (2026-11-14, 10초 기록)", fontsize=13, fontweight="bold")
fig.savefig("results/mic_fft_test.png", dpi=150)
