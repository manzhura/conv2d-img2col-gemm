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
    def forward(ctx, x, w, bias, stride, padding, dilation,
                BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
                NUM_WARPS=4, NUM_STAGES=2,
                ):
        assert x.is_cuda and w.is_cuda
        N, Cin, H, W_ = x.shape
        Cout, Cin_w, Kh, Kw = w.shape
        assert Cin == Cin_w
        Sh, Sw = stride; Ph, Pw = padding; Dh, Dw = dilation

        Ho, Wo = TritonConv2dFn._out_hw(H, W_, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw)
        M = N * Ho * Wo
        K = Cin * Kh * Kw

        # ---- img2col -> cols[M,K] ----
        cols_dtype = torch.float16
        cols = torch.empty((M, K), device=x.device, dtype=cols_dtype)
        sN, sC, sH, sW = x.stride()
        grid_i2c = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        img2col_kernel[grid_i2c](
            x, cols,
            N, Cin, H, W_,
            Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
            Ho, Wo,
            sN, sC, sH, sW,
            K,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            CAST_FP16=(cols_dtype == torch.float16),
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )

        # ---- GEMM: [M,K] @ [K,Cout] -> [M,Cout] ----
        W_mat = w.view(Cout, -1).t().contiguous()
        y_col = triton_gemm(
            cols, W_mat,
            use_fp16=(cols_dtype == torch.float16),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES
        )  # fp32 output

        if bias is not None:
            y_col.add_(bias.float().view(1, -1))

        y = y_col.view(N, Ho, Wo, Cout).permute(0, 3, 1, 2).contiguous()

        # save ctx
        ctx.save_for_backward(x, w, (bias if bias is not None else torch.tensor([], device=x.device, dtype=x.dtype)))
        ctx.has_bias = (bias is not None)
        ctx.stride   = stride
        ctx.padding  = padding
        ctx.dilation = dilation
        ctx.shape_io = (N, Cin, H, W_, Cout, Kh, Kw, Ho, Wo)
        ctx.blocks   = (BLOCK_M, BLOCK_N, BLOCK_K)
        ctx.launch   = (NUM_WARPS, NUM_STAGES)
        return y.to(x.dtype, copy=False)

