# ============================================================
#     TritonConv2dInt8Fn — FULL WORKING FILE (FINAL VERSION)
# ============================================================

import torch
import triton
from torch.autograd import Function

from conv_gemm.triton_kernels.int8.img2col_int8_kernel import img2col_int8
from conv_gemm.triton_kernels.int8.gemm_int8_kernel  import gemm_int8_tc
from conv_gemm.triton_kernels.int8.col2img_int8_kernel import col2img_int32
from conv_gemm.triton_kernels.int8.int8_quant import quantize_int8_sym_tensor
from conv_gemm.triton_kernels.fp32.img2col_kernel import img2col_kernel
from conv_gemm.triton_kernels.fp32.col2img_kernel import col2img_kernel
from conv_gemm.triton_kernels.fp32.gemm_kernel import triton_gemm
from conv_gemm.configs.kernel_config import INT8_C2I_CFG, INT8_GEMM_CFG, INT8_I2C_CFG


class TritonConv2dInt8Fn(Function):
    """
    Автоgrad-обёртка для INT8-свёртки на Triton.
    Forward:
    • Ожидает x_q и w_q в int8 (уже квантованные).
    • img2col_int8 → INT8 GEMM (gemm_int8_tc) → INT32 akk → dequant в FP32.
    • Поддерживает padding K до кратности 4 для Tensor Core.
    • Возвращает FP32 feature map [N, Cout, Ho, Wo].
    Backward:
    • Не реализован (функция используется только для инференса INT8).
    """
    @staticmethod
    def forward(ctx, x_q, w_q, w_scale, bias,
                stride, padding, dilation, act_scale
                ):

        assert x_q.dtype == torch.int8
        assert w_q.dtype == torch.int8

        N, Cin, H, W = x_q.shape
        Cout, Cin2, Kh, Kw = w_q.shape
        assert Cin == Cin2

        Sh, Sw = stride
        Ph, Pw = padding
        Dh, Dw = dilation

        Ho = (H + 2 * Ph - Dh * (Kh - 1) - 1) // Sh + 1
        Wo = (W + 2 * Pw - Dw * (Kw - 1) - 1) // Sw + 1
        M = N * Ho * Wo

        K_real = Cin * Kh * Kw
        K_pad = ((K_real + 3) // 4) * 4

        # img2col -> cols[M,K]
        cols_q, _ = img2col_int8(
            x_q,
            Kh, Kw,
            Sh, Sw,
            Ph, Pw,
            Dh, Dw,
            K_pad,
            INT8_I2C_CFG.BLOCK_M,
            INT8_I2C_CFG.BLOCK_K,
            INT8_I2C_CFG.NUM_WARPS,
            INT8_I2C_STAGES.NUM_STAGES,
        )

        Wmat = w_q.view(Cout, K_real).t().contiguous()

        if K_pad != K_real:
            Wmat = torch.nn.functional.pad(Wmat, (0, 0, 0, K_pad - K_real))

        # GEMM[M, K] @ [K, Cout] -> [M, Cout]
        Y_i32 = gemm_int8_tc(
            cols_q,
            Wmat,
            INT8_GEMM_CFG.BLOCK_M,
            INT8_GEMM_CFG.BLOCK_N,
            INT8_GEMM_CFG.BLOCK_K,
            INT8_GEMM_CFG.NUM_WARPS,
            INT8_GEMM_CFG.NUM_STAGES,
        )

        # Dequantization
        deq = float(w_scale) * float(act_scale)
        y_fp32 = Y_i32.float() * deq
        if bias is not None:
            y_fp32 += bias.float()
        y_fp32 = y_fp32.view(N, Ho, Wo, Cout).permute(0, 3, 1, 2).contiguous()

        return y_fp32


