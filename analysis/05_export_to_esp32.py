# %% [markdown]
# # 05. 학습한 인공신경망을 ESP32에 넣기
# 03번에서 학습한 상태 분류 MLP(입력 8 → 16 → 8 → 출력 3)의 가중치를 C 헤더 파일(`model_weights.h`)로 바꾼다. 펌웨어에서 `USE_MLP 1`로 바꿔 업로드하면 신호등 색을 **인공신경망이 직접 판단**한다.
#
# **검증:** 장치의 C 코드와 똑같은 계산(z-점수 → 행렬곱 → ReLU → 행렬곱 → ReLU → 행렬곱 → 가장 큰 값)을 32비트 실수로 다시 해서, 파이썬 모델의 예측과 **모든 기록에서 같은지** 확인한다.

# %%
import os, sys, numpy as np, pandas as pd, joblib
from common import *
sys.path.insert(0, os.path.join(ROOT, "tools"))
from export_mlp import FEATURES

pack = joblib.load(f"{PROC}/state_mlp8.joblib"); m, sc = pack["model"], pack["scaler"]
assert [f[2:] for f in pack["feats"]] == FEATURES, "특징 순서가 펌웨어와 다름"
W = [np.asarray(w, np.float64) for w in m.coefs_]; B = [np.asarray(b, np.float64) for b in m.intercepts_]

# 펌웨어 출력은 항상 3칸(0=최적, 1=졸림, 2=산만). 학습에 없던 상태는 절대 고르지 않도록 편향을 아주 작게 둔다.
cls = list(m.classes_)
if len(cls) == 2 and W[2].shape[1] == 1:          # 이진 분류: 출력 1개(로짓) → 두 칸으로 펼침
    W[2] = np.hstack([-W[2] / 2, W[2] / 2]); B[2] = np.array([-B[2][0] / 2, B[2][0] / 2])
W3 = np.zeros((W[2].shape[0], 3)); B3 = np.full(3, -1e4)
for j, c in enumerate(cls): W3[:, c] = W[2][:, j]; B3[c] = B[2][j]
W[2], B[2] = W3, B3
print("학습된 상태:", [STATE_KO[c] for c in cls], "| 층 크기:", [w.shape for w in W])

# %%
def cf(x):                                        # C 실수 표기 (정수 모양이면 .0 을 붙여야 f 접미사가 유효)
    s = f"{x:.7g}"
    return (s if any(ch in s for ch in ".en") else s + ".0") + "f"
def arr1(n, v): return f"const float {n}[{len(v)}] = {{{', '.join(cf(x) for x in v)}}};\n"
def arr2(n, M):
    rows = ",\n  ".join("{" + ", ".join(cf(x) for x in r) + "}" for r in M)
    return f"const float {n}[{M.shape[0]}][{M.shape[1]}] = {{\n  {rows}\n}};\n"
std = np.where(sc.scale_ == 0, 1.0, sc.scale_)
path = os.path.join(ROOT, "firmware/classroom_env/model_weights.h")
with open(path, "w", encoding="utf-8") as f:
    f.write("// 자동 생성 파일 — analysis/05_export_to_esp32.py\n// 입력 순서: " + ", ".join(FEATURES) + "\n#pragma once\n")
    f.write(f"#define MLP_IN {W[0].shape[0]}\n#define MLP_H1 {W[0].shape[1]}\n#define MLP_H2 {W[1].shape[1]}\n#define MLP_OUT 3\n")
    f.write(arr1("MLP_MEAN", sc.mean_) + arr1("MLP_STD", std))
    for k in range(3): f.write(arr2(f"MLP_W{k+1}", W[k].T) + arr1(f"MLP_B{k+1}", B[k]))
print("저장:", path, f"({os.path.getsize(path)/1024:.1f} KB)")

# %% [markdown]
# ## 검증: 장치 계산 = 파이썬 모델?

# %%
lab = pd.read_csv(f"{PROC}/labeled.csv", encoding="utf-8-sig").dropna(subset=pack["feats"])
X = lab[pack["feats"]].to_numpy()
f32 = np.float32
z = ((X - sc.mean_) / std).astype(f32)
h = np.maximum(0, z @ W[0].astype(f32) + B[0].astype(f32))
h = np.maximum(0, h @ W[1].astype(f32) + B[1].astype(f32))
dev = np.argmax(h @ W[2].astype(f32) + B[2].astype(f32), axis=1)
py = m.predict(sc.transform(lab[pack["feats"]]))
print(f"비교한 기록 {len(X):,}개 중 예측이 같은 기록: {(dev == py).sum():,}개 ({(dev == py).mean()*100:.2f}%)")
