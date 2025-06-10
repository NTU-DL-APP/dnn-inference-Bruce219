import numpy as np
import json

# === Activation functions ===
def relu(x):
    """ReLU: max(0, x)"""
    return np.maximum(0, x)

def softmax(x):
    """Softmax with numerical-stability trick"""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# 把支援的 activation 做個對照表，較易擴充／管理
_ACT_FUNCS = {
    "relu": relu,
    "softmax": softmax,
}

def _apply_activation(x, act_name):
    """依名稱呼叫對應 activation；若名稱是 None 則直接回傳 x。"""
    if act_name is None:
        return x
    try:
        return _ACT_FUNCS[act_name](x)
    except KeyError:
        raise ValueError(f"Unsupported activation: {act_name}")

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# === Model Forward ===
# 支援層類型：Dense、Flatten、Activation（relu, softmax）
def nn_forward_h5(model_arch, weights, data):
    x = data.astype(np.float32)  # 確保輸入為 float32，可視需要保留／刪除

    for layer in model_arch:
        ltype   = layer["type"]
        cfg     = layer.get("config", {})
        wnames  = layer.get("weights", [])

        # 1. Flatten
        if ltype == "Flatten":
            x = flatten(x)

        # 2. Dense (+ 內建 activation)
        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            x = _apply_activation(x, cfg.get("activation"))

        # 3. 獨立 Activation 層
        elif ltype == "Activation":
            x = _apply_activation(x, cfg.get("activation"))

        # 4. 其他層型別尚未支援
        else:
            raise ValueError(f"Unsupported layer type: {ltype}")

    return x
