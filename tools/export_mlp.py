"""
학습된 scikit-learn MLPClassifier 를 ESP32 용 C 헤더(model_weights.h)로 내보낸다.

사용 예 (코랩/주피터에서 모델 학습 후):
    from export_mlp import export_mlp
    export_mlp(model, scaler, "firmware/classroom_env/model_weights.h")

조건
- 입력 특징 8개, 순서 고정:
  snd_mean, snd_std, snd_spikes, co2_mean, co2_slope, lux, temp, hum
- 은닉층 2개, 활성화 함수 ReLU (MLPClassifier(hidden_layer_sizes=(h1, h2), activation="relu"))
- 출력 클래스 0=최적, 1=졸림, 2=산만 (model.classes_ 가 [0, 1, 2] 여야 함)
- scaler 는 StandardScaler (z-점수 표준화)
그 뒤 config.h 에서 USE_MLP 를 1 로 바꾸고 업로드하면 된다.
"""
import numpy as np

FEATURES = ["snd_mean", "snd_std", "snd_spikes", "co2_mean",
            "co2_slope", "lux", "temp", "hum"]


def _arr1(name, v):
    vals = ", ".join(f"{x:.7g}f" for x in v)
    return f"const float {name}[{len(v)}] = {{{vals}}};\n"


def _arr2(name, m):
    rows = ",\n  ".join("{" + ", ".join(f"{x:.7g}f" for x in r) + "}" for r in m)
    return f"const float {name}[{m.shape[0]}][{m.shape[1]}] = {{\n  {rows}\n}};\n"


def export_mlp(model, scaler, path="model_weights.h"):
    if len(model.coefs_) != 3:
        raise ValueError("은닉층이 정확히 2개인 모델만 지원합니다.")
    if model.activation != "relu":
        raise ValueError("activation='relu' 모델만 지원합니다.")
    if list(model.classes_) != [0, 1, 2]:
        raise ValueError(f"클래스가 [0,1,2] 여야 합니다. 현재: {list(model.classes_)}")
    W1, W2, W3 = (np.asarray(w).T for w in model.coefs_)   # sklearn 은 (입력, 출력) → (출력, 입력)
    b1, b2, b3 = (np.asarray(b) for b in model.intercepts_)
    if W1.shape[1] != len(FEATURES):
        raise ValueError(f"입력 특징이 {len(FEATURES)}개여야 합니다. 현재: {W1.shape[1]}")
    std = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)

    out = ["// 자동 생성 파일 — tools/export_mlp.py\n",
           "// 입력 순서: " + ", ".join(FEATURES) + "\n",
           "#pragma once\n",
           f"#define MLP_IN {W1.shape[1]}\n#define MLP_H1 {W1.shape[0]}\n",
           f"#define MLP_H2 {W2.shape[0]}\n#define MLP_OUT {W3.shape[0]}\n",
           _arr1("MLP_MEAN", scaler.mean_), _arr1("MLP_STD", std),
           _arr2("MLP_W1", W1), _arr1("MLP_B1", b1),
           _arr2("MLP_W2", W2), _arr1("MLP_B2", b2),
           _arr2("MLP_W3", W3), _arr1("MLP_B3", b3)]
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)
    print(f"저장 완료: {path}  ({W1.shape[1]}-{W1.shape[0]}-{W2.shape[0]}-{W3.shape[0]})")


def c_forward(path_free_model, scaler, x):
    """ESP32 코드와 똑같은 계산을 파이썬으로 재현 — 내보내기 검증용."""
    m = path_free_model
    z = (np.asarray(x, float) - scaler.mean_) / np.where(scaler.scale_ == 0, 1, scaler.scale_)
    h = np.maximum(0, z @ m.coefs_[0] + m.intercepts_[0])
    h = np.maximum(0, h @ m.coefs_[1] + m.intercepts_[1])
    return int(np.argmax(h @ m.coefs_[2] + m.intercepts_[2]))
