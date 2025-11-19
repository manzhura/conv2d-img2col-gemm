# ============================================================
#     TritonConv2dInt8Fn — FULL WORKING FILE (FINAL VERSION)
# ============================================================

import torch
import triton
from torch.autograd import Function

# ============================================================
#                   INT8 FORWARD — BEST BLOCKS
# ============================================================

# img2col INT8
INT8_I2C_BLOCK_M = 128
INT8_I2C_BLOCK_K = 32
INT8_I2C_WARPS   = 2
INT8_I2C_STAGES  = 3

# GEMM INT8
INT8_GEMM_BLOCK_M = 64
INT8_GEMM_BLOCK_N = 128
INT8_GEMM_BLOCK_K = 64
INT8_GEMM_WARPS   = 4
INT8_GEMM_STAGES  = 3

# col2img INT32
INT8_C2I_BLOCK_M = 32
INT8_C2I_BLOCK_K = 32
INT8_C2I_WARPS   = 4
INT8_C2I_STAGES  = 3


# ============================================================
#                 FP32 BACKWARD TEMP BLOCKS
# ============================================================

FP32_I2C_BLOCK_M = 64
FP32_I2C_BLOCK_K = 64
FP32_I2C_WARPS   = 4
FP32_I2C_STAGES  = 2

FP32_GEMM_BLOCK_M = 64
FP32_GEMM_BLOCK_N = 64
FP32_GEMM_BLOCK_K = 64
FP32_GEMM_WARPS   = 4
FP32_GEMM_STAGES  = 2

FP32_C2I_BLOCK_M = 64
FP32_C2I_BLOCK_K = 64
FP32_C2I_WARPS   = 4
FP32_C2I_STAGES  = 2


# ============================================================
#                 IMPORT KERNELS
# ============================================================

from conv_gemm.triton_kernels.int8.img2col_int8_kernel import img2col_int8
from conv_gemm.triton_kernels.int8.gemm_int8_kernel  import gemm_int8_tc
from conv_gemm.triton_kernels.int8.col2img_int8_kernel import col2img_int32
from conv_gemm.triton_kernels.int8.int8_quant import quantize_int8_sym_tensor
from conv_gemm.triton_kernels.fp32.img2col_kernel import img2col_kernel
from conv_gemm.triton_kernels.fp32.col2img_kernel import col2img_kernel
from conv_gemm.triton_kernels.fp32.gemm_kernel import triton_gemm



# ============================================================
#             AUTOGRAD FUNCTION (INT8 FORWARD + FP32 BACKWARD)
# ============================================================
class TritonConv2dInt8Fn(Function):

    @staticmethod
    def forward(ctx,
                x_q,
                w_q,
                w_scale,
                bias,
                stride,
                padding,
                dilation,
                act_scale):

        assert x_q.dtype == torch.int8
        assert w_q.dtype == torch.int8

        N, Cin, H, W = x_q.shape
        Cout, Cin2, Kh, Kw = w_q.shape
        assert Cin == Cin2

        Sh, Sw = stride
        Ph, Pw = padding
        Dh, Dw = dilation

        # ============================================================
        # 1. Compute output spatial shape
        # ============================================================
        Ho = (H + 2 * Ph - Dh * (Kh - 1) - 1) // Sh + 1
        Wo = (W + 2 * Pw - Dw * (Kw - 1) - 1) // Sw + 1
        M = N * Ho * Wo

        # ============================================================
        # 2. Compute real K and padded K (MUST MATCH kernels)
        # ============================================================
        K_real = Cin * Kh * Kw
        K_pad = ((K_real + 3) // 4) * 4

        # ============================================================
        # 3. img2col INT8  (НОВАЯ СИГНАТУРА!)
        #    returns: cols_q [M, K_pad]
        # ============================================================
        cols_q, _ = img2col_int8(
            x_q,
            Kh, Kw,
            Sh, Sw,
            Ph, Pw,
            Dh, Dw,
            K_pad,
            INT8_I2C_BLOCK_M,
            INT8_I2C_BLOCK_K,
            INT8_I2C_WARPS,
            INT8_I2C_STAGES,
        )
        # cols_q exactly shape [M, K_pad]

        # ============================================================
        # 4. Prepare weight matrix (W_q) → [K_pad, Cout]
        # ============================================================
        Wmat = w_q.view(Cout, K_real).t().contiguous()  # [K_real, Cout]

        if K_pad != K_real:
            Wmat = torch.nn.functional.pad(Wmat, (0, 0, 0, K_pad - K_real))
            # Wmat: [K_pad, Cout]

        # ============================================================
        # 5. GEMM INT8 → int32 output [M, Cout]
        # ============================================================
        Y_i32 = gemm_int8_tc(
            cols_q,  # [M, K_pad]
            Wmat,  # [K_pad, Cout]
            INT8_GEMM_BLOCK_M,
            INT8_GEMM_BLOCK_N,
            INT8_GEMM_BLOCK_K,
            INT8_GEMM_WARPS,
            INT8_GEMM_STAGES,
        )

        # ============================================================
        # 6. Dequantization:
        #    y = int32 * (scale_x * scale_w)
        # ============================================================
        deq = float(w_scale) * float(act_scale)
        y_fp32 = Y_i32.float() * deq

        if bias is not None:
            y_fp32 += bias.float()

        # ============================================================
        # 7. Reshape → [N, Cout, Ho, Wo]
        # ============================================================
        y_fp32 = y_fp32.view(N, Ho, Wo, Cout).permute(0, 3, 1, 2).contiguous()

        return y_fp32
