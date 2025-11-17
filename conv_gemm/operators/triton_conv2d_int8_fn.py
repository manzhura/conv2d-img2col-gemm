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
    def forward(ctx, x, w, bias, stride, padding, dilation):
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
        w_q, s_w = quantize_int8_sym(w)

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
        W_q_mat_raw = w_q.view(Cout, -1).t().contiguous()   # [K, Cout]
        if K_pad != K:
            W_q_mat = torch.nn.functional.pad(
                W_q_mat_raw, (0, 0, 0, K_pad - K)
            )
        else:
            W_q_mat = W_q_mat_raw

        # ----------- INT8 GEMM -----------
        Y_i32 = gemm_int8_tc(
            cols_q,
            W_q_mat,
            BLOCK_M=INT8_GEMM_BLOCK_M,
            BLOCK_N=INT8_GEMM_BLOCK_N,
            BLOCK_K=INT8_GEMM_BLOCK_K,
            num_warps=INT8_GEMM_WARPS,
            num_stages=INT8_GEMM_STAGES,
        )  # [M, Cout], int32

        # ----------- DEQUANT + BIAS -----------
        y_fp32 = Y_i32.float() * (s_x * s_w)
        if bias is not None:
            y_fp32 += bias.float().view(1, -1)

        y = y_fp32.view(N, Ho, Wo, Cout).permute(0, 3, 1, 2).contiguous()

        # ----------- SAVE FOR BACKWARD -----------
        ctx.save_for_backward(x, w, bias)
        ctx.shape_io = (N, Cin, H, W, Cout, Kh, Kw, Ho, Wo)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return y.to(x.dtype, copy=False)


    # ============================================================
    #                        BACKWARD
    # ============================================================
    @staticmethod
    def backward(ctx, gy):
        x, w, bias = ctx.saved_tensors
        (N, Cin, H, W, Cout, Kh, Kw, Ho, Wo) = ctx.shape_io
        Sh, Sw = ctx.stride
        Ph, Pw = ctx.padding
        Dh, Dw = ctx.dilation

        M = N * Ho * Wo
        K = Cin * Kh * Kw

        gy32 = gy.float()
        x32  = x.float()
        w32  = w.float()

        # --------- dBias --------
        gb = gy32.sum((0, 2, 3)) if bias is not None else None

        # ============================================================
        #                              dW
        # ============================================================
        cols = torch.empty((M, K), device=x.device, dtype=torch.float32)

        grid_i2c = (
            triton.cdiv(M, FP32_I2C_BLOCK_M),
            triton.cdiv(K, FP32_I2C_BLOCK_K),
        )

        img2col_kernel[grid_i2c](
            x32, cols,
            N, Cin, H, W,
            Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
            Ho, Wo,
            *x32.stride(),
            K,
            BLOCK_M=FP32_I2C_BLOCK_M,
            BLOCK_K=FP32_I2C_BLOCK_K,
            CAST_FP16=False,
            num_warps=FP32_I2C_WARPS,
            num_stages=FP32_I2C_STAGES,
        )

        dy_mat = gy32.permute(0, 2, 3, 1).reshape(M, Cout)
        cols_T = cols.t().contiguous()

        dW_mat = triton_gemm(
            cols_T, dy_mat,
            use_fp16=False,
            BLOCK_M=FP32_GEMM_BLOCK_M,
            BLOCK_N=FP32_GEMM_BLOCK_N,
            BLOCK_K=FP32_GEMM_BLOCK_K,
            num_warps=FP32_GEMM_WARPS,
            num_stages=FP32_GEMM_STAGES,
        )

        gw = dW_mat.view(Cin, Kh, Kw, Cout).permute(3, 0, 1, 2).contiguous()

        # ============================================================
        #                               dX
        # ============================================================
        W_matT = w32.view(Cout, -1)

        dcols = triton_gemm(
            dy_mat, W_matT,
            use_fp16=False,
            BLOCK_M=FP32_GEMM_BLOCK_M,
            BLOCK_N=FP32_GEMM_BLOCK_K,
            BLOCK_K=Cout,
            num_warps=FP32_GEMM_WARPS,
            num_stages=FP32_GEMM_STAGES,
        )

        dx32 = torch.zeros((N, Cin, H, W), device=x.device, dtype=torch.float32)

        grid_c2i = (
            triton.cdiv(M, FP32_C2I_BLOCK_M),
            triton.cdiv(K, FP32_C2I_BLOCK_K),
        )

        col2img_kernel[grid_c2i](
            dcols, dx32,
            N, Cin, H, W,
            Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
            Ho, Wo,
            *dx32.stride(),
            K,
            BLOCK_M=FP32_C2I_BLOCK_M,
            BLOCK_K=FP32_C2I_BLOCK_K,
            num_warps=FP32_C2I_WARPS,
            num_stages=FP32_C2I_STAGES,
        )

        return (
            dx32.to(x.dtype, copy=False),
            gw.to(w.dtype, copy=False),
            gb.to(bias.dtype, copy=False) if bias is not None else None,
            None, None, None
        )
