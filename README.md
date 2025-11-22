# Conv2d Reimagined: img2col, GEMM, Sparsity & INT8 Quantization

Проект исследует реализацию Conv2d через разложение:
**img2col → GEMM → col2img**  
и анализирует влияние **квантизации (INT8)** и **спарсификации** на производительность.

Реализованы собственные Triton-ядра для FP16 и INT8, собственные Conv2d-слои, PTQ-квантование, forward/backward для FP16, INT8-inference, а так же бенчмарк для них.

## Возможности

### FP16 (полная поддержка)
- FP16 img2col
- FP16 GEMM
- FP16 col2img
- TritonConv2d: forward (FP16) + backward (FP32)
- Sparsity:
  - input-channel sparsity
  - output-channel sparsity
  - block sparsity
  - отдельные маски для forward/backward

### INT8 (inference-only)
- INT8 img2col
- INT8 GEMM
- INT32 col2img 
- TritonConv2d: forward (INT8)
- PTQ-квантование:
  - symmetric per-tensor scale
  - квантованные веса (int8)
  - bias — FP32

### Benchmarking
- Сравнение PyTorch Conv2d (FP16) и Triton FP16/INT8
- Перебор сетки параметров:
  N, Cin, Cout, H, K, S, P
- Метрики:
  - forward time
  - backward time (FP16)
  - speedup
  - MAE / max error
  - выделенная память
- Сохранение всех результатов в CSV

## Установка

```bash
git https://github.com/manzhura/conv2d-img2col-gemm.git
cd conv2d-img2col-gemm

python3 -m venv venv
source venv/bin/activate          

pip install -r requirements.txt
```

## Структура проекта

```
conv2d-img2col-gemm/
  conv_gemm/
    baseline_layers/
      triton_conv2d.py
      triton_conv2d_int8.py
    baseline_operators/
      triton_conv2d_fp16_fn.py
      triton_conv2d_int8_fn.py
    configs/
      kernel_config.py
    triton_kernels/
      fp16/
        img2col_kernel.py
        gemm_kernel.py
        col2img_kernel.py
      int8/
        img2col_int8_kernel.py
        gemm_int8_kernel.py
        col2img_int8_kernel.py
        int8_quant.py
  notebooks/
  data/
  README.md
  requirements.txt
```