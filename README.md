<img width="1024" height="1024" alt="ChatGPT Image 24 Nov 2025, 14 43 37" src="https://github.com/user-attachments/assets/eadf7202-1325-4a6f-86ee-6176b5717ad0" />

# pqnt — Power-Law Quantization Engine  
C++ Core + Pybind11 Python API (Source Build Edition)

`pqnt` is a high-accuracy **post-training quantization engine** implemented in modern C++17 and exposed to Python via Pybind11.

This README explains:
- How quantization works
- The mathematics behind PQNT
- How to build the project from source (Linux, FreeBSD, macOS)
- How to test & integrate PQNT with PyTorch/NumPy

---

# 🔍 1. What is Quantization?

Quantization converts FP32 values into INT8 to reduce:

| Impact | Reason |
|--------|--------|
| Model size ↓ 4× | 32-bit → 8-bit |
| Inference latency ↓ | Cheaper int8 ops |
| Memory bandwidth ↓ | Crucial for mobile/edge |
| Energy consumption ↓ | Smaller activations |

---

# ⚠️ 2. Why Do Standard Methods Fail?

Typical PTQ (MinMax, Percentile, StdClip) struggle because distributions are:

- Non-Gaussian  
- Heavy-tailed  
- Channel-asymmetric  
- Sensitive to outliers  

This causes:
- Wrong scaling  
- Saturated quantization  
- High output mismatch  
- Accuracy collapse  

---

# 🌟 3. PQNT — Power-Transform Quantization

PQNT introduces a **pre-quantization non-linear shaping** to compress extreme values before converting to INT8.

### ✔ 3.1 Power Transform

\[
t = \text{sign}(x) \cdot |x|^p
\]

For \( p < 1 \):  
- Outliers shrink  
- Distribution becomes smoother  
- Quantization becomes easier

### ✔ 3.2 INT8 Symmetric Quantization

\[
s = \frac{\max(|t|)}{127}
\]
\[
q = \text{clip}\left( \left\lfloor \frac{t}{s} \right\rceil, -128, 127 \right)
\]

### ✔ 3.3 Dequantization

\[
\tilde{t} = q \cdot s
\]

### ✔ 3.4 Inverse Transform

\[
\hat{x} = \text{sign}(\tilde{t}) \cdot |\tilde{t}|^{1/p}
\]

This returns values near their original domain.

---

# 🧪 4. Benchmarking Results

Across CNN/Transformer families (ResNet, MobileNet, ViT, Swin):

- PQNT consistently gives **lowest per-layer MSE**
- PQNT yields **best logit MSE vs FP32**
- PQNT maintains good accuracy without calibration
- PQNT outperforms MinMax, Percentile, ACIQ, KLD, SmoothQuant-lite

---
<img width="847" height="299" alt="image" src="https://github.com/user-attachments/assets/6b31605c-5e0f-42b7-8cd7-4e6b08722e9f" />
<img width="1151" height="700" alt="image" src="https://github.com/user-attachments/assets/2865c1b3-06cc-47d5-9225-29862ffd0f0e" />
<img width="979" height="547" alt="image" src="https://github.com/user-attachments/assets/fc8e2d96-38ce-4935-9d34-3e01bd9ff055" />
<img width="1160" height="602" alt="image" src="https://github.com/user-attachments/assets/3c810474-0033-4937-b723-3b49f66d0b53" />

# 🛠 5. Build From Source

## 5.1 Requirements

### Linux / macOS / FreeBSD
- Python ≥ 3.8  
- CMake ≥ 3.15  
- Ninja (recommended)
- A modern C++17 compiler (`g++`, `clang`, or FreeBSD's `cc`)
- pybind11  


Install dependencies:

```
pip install scikit-build-core pybind11 numpy
```

---

## 5.2 Manual CMake Build
5.2.1. Clone Repository
```
git clone https://github.com/Michael-Obs66/pqnt

```
5.2.2. Install External in external folder
```
bash setup_external.sh

```
5.2.3. Make "build" directory and enter to directory
```
mkdir build
cd build
```
5.2.4. Make/Compile
```
cmake .. 
make -j4
```

Outputs:
- `pqnt_cli` — CLI tester
- `pqnt*.so` — Python extension module

---

# 📘 6. Example Usage in Python

```python
import numpy as np
import pqnt

x = np.random.randn(1024).astype(np.float32)

q, scale, p = pqnt.quantize_tensor(x, p=0.56)
xhat = pqnt.dequant_tensor(q, scale, p)
```

Per-channel:

```python
x4d = np.random.randn(1, 64, 56, 56).astype(np.float32)
out = pqnt.quantize_tensor_per_channel(x4d, axis=1, p=0.56)
```

---

# 📘 7. Example Usage in Python using RESNET50
```python
import torch
import torchvision.models as models
import numpy as np
import pqnt   # import library PQNT

# -------------------------------------------------------
# 1. Load ResNet50 FP32
# -------------------------------------------------------
model = models.resnet50(weights="IMAGENET1K_V1")
model.eval()

# -------------------------------------------------------
# 2. Helper Function: quantize weight tensor using PQNT
# -------------------------------------------------------
def quantize_weights_pqnt(tensor, p=0.56):
    """Quantize 1 tensor (NumPy array) menggunakan PQNT."""
    w = tensor.detach().cpu().numpy().astype(np.float32)

    # quantize (INT8) + dequant (float32 reconstructed)
    q, scale, pval = pqnt.quantize_tensor(w, p)

    # convert kembali ke torch
    w_hat = torch.from_numpy(q.astype(np.float32) * scale)
    return w_hat.to(tensor.device)

# -------------------------------------------------------
# 3. Apply quantization (weight-only per tensor)
# -------------------------------------------------------
for name, param in model.named_parameters():
    if param.ndim >= 2:       # if only weight conv/linear
        param.data = quantize_weights_pqnt(param, p=0.56)

# -------------------------------------------------------
# 4. Inference Test
# -------------------------------------------------------
x = torch.randn(1, 3, 224, 224)  # dummy image
with torch.no_grad():
    out = model(x)

print("Output model:", out[0][:5])

```
---

# 📂 8. Project Layout

```
pqnt/
  ├── external/setup_external.sh
  ├── include/pqnt.hpp
  ├── src/bindings.cpp
  ├── src/bindings.cpp
  ├── src/main.cpp
  ├── src/pqnt.cpp
  .gitignore
  CMakeLists.txt
  LISENCE
  MANIFEST.in
  __init__.py
  pyproject.toml
  setup.py
```

---

# 📄 License

MIT License.

---

# 🤝 Contributing

Pull requests and benchmarking reports are welcome.

