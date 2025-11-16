# conv_gemm/operators/triton_conv2d_fp32_fn.py

import torch
import triton

from conv_gemm.triton_kernels.fp32.img2col_kernel import img2col_kernel
from conv_gemm.triton_kernels.fp32.col2img_kernel import col2img_kernel
from conv_gemm.triton_kernels.fp32.gemm_kernel import triton_gemm


class TritonConv2dFn(torch.autograd.Function):
    @staticmethod
    def _out_hw(H, W, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw):
        Ho = (H + 2 * Ph - Dh * (Kh - 1) - 1) // Sh + 1
        Wo = (W + 2 * Pw - Dw * (Kw - 1) - 1) // Sw + 1
        return Ho, Wo

    @staticmethod
    def forward(
            ctx, x, w, bias,
            stride, padding, dilation,
            BLOCK_M, BLOCK_N, BLOCK_K,
            NUM_WARPS, NUM_STAGES,
            I2C_FP16, GEMM_FP16
    ):
        N, Cin, H, W = x.shape
        Cout, CinW, Kh, Kw = w.shape
        Sh, Sw = stride
        Ph, Pw = padding
        Dh, Dw = dilation
        Ho, Wo = TritonConv2dFn._out_hw(H, W, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw)

        M = N * Ho * Wo
        K = Cin * Kh * Kw

        # cols buffer
        cols_dtype = torch.float16 if (I2C_FP16 or GEMM_FP16) else torch.float32
        cols = torch.empty((M, K), device=x.device, dtype=cols_dtype)

        sN, sC, sH, sW = x.stride()
        grid_i2c = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

        img2col_kernel[grid_i2c](
            x, cols,
            N, Cin, H, W,
            Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
            Ho, Wo,
            sN, sC, sH, sW,
            K,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            CAST_FP16=(cols_dtype == torch.float16),
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        # GEMM
        W_mat = w.view(Cout, -1).t().contiguous()
        y_col = triton_gemm(
            cols, W_mat,
            use_fp16=(cols_dtype == torch.float16),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES
        )

        if bias is not None:
            y_col += bias.float().view(1, -1)

        y = y_col.view(N, Ho, Wo, Cout).permute(0, 3, 1, 2).contiguous()

        # save ctx for backward
        ctx.save_for_backward(x, w, bias if bias is not None else torch.tensor([], device=x.device))
        ctx.has_bias = bias is not None
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.shape_io = (N, Cin, H, W, Cout, Kh, Kw, Ho, Wo)
        ctx.blocks = (BLOCK_M, BLOCK_N, BLOCK_K)
        ctx.launch = (NUM_WARPS, NUM_STAGES)

        return y.to(x.dtype, copy=False)
