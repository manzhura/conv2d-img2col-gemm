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
from conv_gemm.triton_kernels.int8.int8_quant import quantize_int8_sym
from conv_gemm.triton_kernels.fp32.img2col_kernel import img2col_kernel
from conv_gemm.triton_kernels.fp32.col2img_kernel import col2img_kernel
from conv_gemm.triton_kernels.fp32.gemm_kernel import triton_gemm



# ============================================================
#             AUTOGRAD FUNCTION (INT8 FORWARD + FP32 BACKWARD)
# ============================================================

class TritonConv2dInt8Fn(Function):

    @staticmethod
    def _out_hw(H, W, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw):
        Ho = (H + 2*Ph - Dh*(Kh-1) - 1)//Sh + 1
        Wo = (W + 2*Pw - Dw*(Kw-1) - 1)//Sw + 1
        return Ho, Wo


    # ============================================================
    #                         FORWARD
    # ============================================================
    @staticmethod
    def forward(ctx, x, w, w_scale, bias, stride, padding, dilation):
        assert x.is_cuda and w.is_cuda

        N, Cin, H, W = x.shape
        Cout, Cinw, Kh, Kw = w.shape
        assert Cin == Cinw

        Sh, Sw = stride
        Ph, Pw = padding
        Dh, Dw = dilation

        Ho, Wo = TritonConv2dInt8Fn._out_hw(H, W, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw)

        M = N * Ho * Wo
        K = Cin * Kh * Kw
        K_pad = ((K + 3) // 4) * 4   # align to 4

        # ----------- Quantize -----------
        x_q, s_x = quantize_int8_sym(x)

        # ----------- INT8 img2col -----------
        cols_q, _ = img2col_int8(
            x_q,
            Kh, Kw,
            Sh, Sw,
            Ph, Pw,
            Dh, Dw,
            K_pad,
            BLOCK_M=INT8_I2C_BLOCK_M,
            BLOCK_K=INT8_I2C_BLOCK_K,
            num_warps=INT8_I2C_WARPS,
            num_stages=INT8_I2C_STAGES,
        )   # → [M, K_pad], int8

        # ----------- Pad weights to K_pad -----------
        W_mat_raw = w.view(Cout, -1).t().contiguous()   # [K, Cout]
        if K_pad != K:
            W_mat = torch.nn.functional.pad(
                W_mat_raw, (0, 0, 0, K_pad - K)
            )
        else:
            W_mat = W_mat_raw

        # ----------- INT8 GEMM -----------
        Y_i32 = gemm_int8_tc(
            cols_q,
            W_mat,
            BLOCK_M=INT8_GEMM_BLOCK_M,
            BLOCK_N=INT8_GEMM_BLOCK_N,
            BLOCK_K=INT8_GEMM_BLOCK_K,
            num_warps=INT8_GEMM_WARPS,
            num_stages=INT8_GEMM_STAGES,
        )  # [M, Cout], int32

        # ----------- DEQUANT + BIAS -----------
        y_fp32 = Y_i32.float() * (s_x * w_scale)
        if bias is not None:
            y_fp32 += bias.float().view(1, -1)

        y = y_fp32.view(N, Ho, Wo, Cout).permute(0, 3, 1, 2).contiguous()

        # ----------- SAVE FOR BACKWARD -----------
        ctx.save_for_backward(x, w, w_scale, bias)
        ctx.shape_io = (N, Cin, H, W, Cout, Kh, Kw, Ho, Wo)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return y.to(x.dtype, copy=False)


