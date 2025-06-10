import sys
import json
import numpy as np
from pathlib import Path

# === 模型檔案路徑（可由命令列參數覆寫） ===
if len(sys.argv) >= 3:
    MODEL_WEIGHTS_PATH = sys.argv[1]
    MODEL_ARCH_PATH    = sys.argv[2]
else:                                       # 預設檔名可自行更改
    MODEL_WEIGHTS_PATH = "fashion_mnist.npz"
    MODEL_ARCH_PATH    = "fashion_mnist.json"

# === 載入權重與架構 ===
weights = np.load(MODEL_WEIGHTS_PATH)
with open(MODEL_ARCH_PATH, "r") as f:
    architecture = json.load(f)

# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

# 對照表方便擴充
_ACT_FUNCS = {
    "relu": relu,
    "softmax": softmax,
}

def apply_activation(x, act_name):
    """依名稱套用 activation；若 None 則直接回傳 x。"""
    if act_name is None:
        return x
    if act_name not in _ACT_FUNCS:
        raise ValueError(f"Unsupported activation: {act_name}")
    return _ACT_FUNCS[act_name](x)

# === 基本層 ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

def dense(x, W, b):
    return x @ W + b

# === Forward pass ===
def forward(x):
    """x 需為 NumPy array，shape = (batch, ...)"""
    for layer in architecture:
        ltype  = layer["type"]
        cfg    = layer.get("config", {})
        wnames = layer.get("weights", [])

        if ltype == "Flatten":
            x = flatten(x)

        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            x = apply_activation(x, cfg.get("activation"))

        elif ltype == "Activation":
            x = apply_activation(x, cfg.get("activation"))

        else:
            raise ValueError(f"Unsupported layer type: {ltype}")

    return x

# === Example usage ===
if __name__ == "__main__":
    # 這裡示範一筆隨機輸入 (28×28 灰階圖片，先 flatten 成 784 維向量)
    dummy_input = np.random.rand(1, 28 * 28).astype(np.float32)
    probs = forward(dummy_input)

    print("🧠 Output probabilities:", probs)
    print("✅ Predicted class:", np.argmax(probs, axis=-1))
