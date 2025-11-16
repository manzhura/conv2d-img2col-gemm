import torch
from torch.autograd import Function

# импортируем твои int8/col2img kernels
# поправь пути под свой проект
from conv_gemm.triton_kernels.int8.img2col_int8_kernel import img2col_int8
from conv_gemm.triton_kernels.int8.gemm_int8_kernel  import gemm_int8_tc
from conv_gemm.triton_kernels.int8.col2img_int8_kernel import col2img_int32

# здесь я использую те имена, что ты кидал в сообщениях:
# img2col_int8(x_q, Kh,Kw, Sh,Sw, Ph,Pw, Dh,Dw, BLOCK_M, BLOCK_K, ...)
# gemm_int8_tc(A_q, B_q, BLOCK_M=..., BLOCK_N=..., BLOCK_K=..., ...)
# col2img_int32(cols_i32, N,Cin,H,W, Kh,Kw, Sh,Sw, Ph,Pw, Dh,Dw, ...)

def quantize_int8_sym(x: torch.Tensor):
    """
    Симметричная пер-тензорная квантизация в int8:
      x_q = round(x / s),   x_q ∈ [-128, 127]
      s  = max(|x|) / 127

    Возвращает:
      x_q: int8-тензор той же формы
      s:   scale (torch.float32 scalar на том же девайсе)
    """
    # 1) гарантируем fp32 внутри
    x32 = x.float()

    # 2) максимум по модулю
    max_abs = x32.abs().max()

    # 3) edge-case: полностью нулевой тензор
    if max_abs == 0:
        scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        x_q = torch.zeros_like(x32, dtype=torch.int8)
        return x_q, scale

    # 4) масштаб: max_abs -> 127
    scale = max_abs / 127.0

    # 5) нормализуем, округляем, клэмпим в [-128, 127] и приводим к int8
    x_q = torch.clamp((x32 / scale).round(), -128, 127).to(torch.int8)

    return x_q, scale
class TritonConv2dInt8Fn(Function):
    @staticmethod
    def _out_hw(H, W, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw):
        Ho = (H + 2 * Ph - Dh * (Kh - 1) - 1) // Sh + 1
        Wo = (W + 2 * Pw - Dw * (Kw - 1) - 1) // Sw + 1
        return Ho, Wo

    @staticmethod
    def forward(ctx, x, w, bias,
                stride, padding, dilation,
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
                NUM_WARPS=4, NUM_STAGES=2):

        assert x.is_cuda and w.is_cuda, "x и w должны быть на CUDA"

        N, Cin, H, W_ = x.shape
        Cout, Cin_w, Kh, Kw = w.shape
        assert Cin == Cin_w
        Sh, Sw = stride
        Ph, Pw = padding
        Dh, Dw = dilation

        # выходные spatial-размеры
        Ho, Wo = TritonConv2dInt8Fn._out_hw(H, W_, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw)
        M = N * Ho * Wo
        K = Cin * Kh * Kw

        # K должен делиться на 4 для int8 dot
        assert K % 4 == 0 and BLOCK_K % 4 == 0, \
            f"K={K}, BLOCK_K={BLOCK_K} должны делиться на 4 для INT8 GEMM"

        # ---- квантование входа и весов ----
        x_q, s_x = quantize_int8_sym(x)
        w_q, s_w = quantize_int8_sym(w)

        # ---- img2col_int8: x_q -> cols_q[M,K] ----
        cols_q, (Ho_check, Wo_check) = img2col_int8(
            x_q,
            Kh, Kw,
            Sh, Sw,
            Ph, Pw,
            Dh, Dw,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        assert Ho_check == Ho and Wo_check == Wo

        # ---- GEMM INT8: [M,K] @ [K,Cout] -> [M,Cout] int32 ----
        W_q_mat = w_q.view(Cout, -1).t().contiguous()  # [K,Cout]
        Y_i32 = gemm_int8_tc(
            cols_q, W_q_mat,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )  # [M, Cout] int32

        # ---- dequant + bias ----
        y_fp32 = Y_i32.float() * (s_x * s_w)  # масштаб
        if bias is not None:
            y_fp32.add_(bias.float().view(1, -1))

        y = y_fp32.view(N, Ho, Wo, Cout).permute(0, 3, 1, 2).contiguous()

        # Сохраняем всё, что нужно для backward
        ctx.save_for_backward(x, w)
        ctx.has_bias = bias is not None
        ctx.stride   = stride
        ctx.padding  = padding
        ctx.dilation = dilation
        ctx.shape_io = (N, Cin, H, W_, Cout, Kh, Kw, Ho, Wo)
        ctx.blocks   = (BLOCK_M, BLOCK_N, BLOCK_K)
        ctx.launch   = (NUM_WARPS, NUM_STAGES)
        ctx.scales   = (s_x, s_w)

        # gy будет приходить в том же dtype, что y, поэтому
        return y.to(x.dtype, copy=False)

    @staticmethod
    def backward(ctx, gy):
        """
        Все тяжёлые матрицы считаем через int8-кернелы:
          - dW: (im2col(x)_q)^T @ gy_q
          - dX: col2img_int32( gy_q @ w_q^T )
        """
        x, w = ctx.saved_tensors
        (N, Cin, H, W_, Cout, Kh, Kw, Ho, Wo) = ctx.shape_io
        Sh, Sw = ctx.stride
        Ph, Pw = ctx.padding
        Dh, Dw = ctx.dilation
        BLOCK_M, BLOCK_N, BLOCK_K = ctx.blocks
        NUM_WARPS, NUM_STAGES = ctx.launch
        s_x, s_w = ctx.scales

        device = x.device
        M = N * Ho * Wo
        K = Cin * Kh * Kw

        # всё для градиента считаем в fp32, но ядра — на int8/int32
        gy32 = gy.float()

        # ---- grad по bias ----
        gb = gy32.sum(dim=(0, 2, 3)) if ctx.has_bias else None

        # ---- квантим gy ----
        gy_q, s_gy = quantize_int8_sym(gy32)  # [N,Cout,Ho,Wo]

        # ---- перепаковываем gy_q в [M,Cout] ----
        gy_q_flat = gy_q.permute(0, 2, 3, 1).contiguous().view(M, Cout)  # [M, Cout]

        # ---- dW: (im2col(x)_q)^T @ gy_q ----
        # заново квантим x и делаем img2col_int8
        x_q, _ = quantize_int8_sym(x)  # тот же scale не обязателен, всё равно STE
        cols_q, (Ho2, Wo2) = img2col_int8(
            x_q,
            Kh, Kw,
            Sh, Sw,
            Ph, Pw,
            Dh, Dw,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        assert Ho2 == Ho and Wo2 == Wo

        # cols_q: [M,K], нам нужно [K,M] @ [M,Cout] -> [K,Cout]
        cols_q_T = cols_q.t().contiguous()  # [K,M]

        dW_i32 = gemm_int8_tc(
            cols_q_T,          # [K,M]
            gy_q_flat,         # [M,Cout]
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )  # [K,Cout] int32

        # dequant dW: масштаб s_x (для cols) и s_gy (для gy)
        dW_fp32 = dW_i32.float() * (s_x * s_gy)
        gw = dW_fp32.view(Cin, Kh, Kw, Cout).permute(3, 0, 1, 2).contiguous()

        # ---- dX: col2img_int32( gy_q @ w_q^T ) ----
        w_q, _ = quantize_int8_sym(w)         # [Cout, Cin, Kh, Kw]
        W_q_mat = w_q.view(Cout, -1).t().contiguous()  # [K,Cout]
        W_q_T   = W_q_mat.t().contiguous()             # [Cout,K]

        # gy_q_flat: [M,Cout], W_q_T: [Cout,K] => [M,K]
        dcols_i32 = gemm_int8_tc(
            gy_q_flat,  # [M,Cout]
            W_q_T,      # [Cout,K]
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )  # [M,K] int32

        dx_i32 = col2img_int32(
            dcols_i32,
            N, Cin, H, W_,
            Kh, Kw,
            Sh, Sw,
            Ph, Pw,
            Dh, Dw,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )  # [N,Cin,H,W] int32

        # масштаб для dx: s_gy * s_w
        dx = dx_i32.float() * (s_gy * s_w)

        # привести к исходным dtypes
        dx = dx.to(x.dtype, copy=False)
        gw = gw.to(w.dtype, copy=False)
        if gb is not None:
            gb = gb.to(w.dtype, copy=False)

        # возвращаем градиенты по позиционным аргументам forward:
        # (x, w, bias, stride, padding, dilation, BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES)
        return dx, gw, gb, None, None, None, None, None, None, None, None
