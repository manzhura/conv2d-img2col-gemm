import torch, triton
from conv_gemm.triton_kernels.fp32.col2img_kernel import col2img_kernel
from conv_gemm.triton_kernels.fp32.gemm_kernel import gemm_kernel, triton_gemm
from conv_gemm.triton_kernels.fp32.img2col_kernel import img2col_kernel


# =========================
# Autograd Function
# =========================
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
                I2C_FP16=False, GEMM_FP16=False):
        assert x.is_cuda and w.is_cuda
        N, Cin, H, W_ = x.shape
        Cout, Cin_w, Kh, Kw = w.shape
        assert Cin == Cin_w
        Sh, Sw = stride; Ph, Pw = padding; Dh, Dw = dilation

        Ho, Wo = TritonConv2dFn._out_hw(H, W_, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw)
        M = N * Ho * Wo
        K = Cin * Kh * Kw

        # ---- img2col -> cols[M,K] ----
        cols_dtype = torch.float16 if I2C_FP16 or GEMM_FP16 else torch.float32
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
        ctx.flags    = (I2C_FP16, GEMM_FP16)
        return y.to(x.dtype, copy=False)

    @staticmethod
    def backward(ctx, gy):
        x, w, bias_stub = ctx.saved_tensors
        (N, Cin, H, W_, Cout, Kh, Kw, Ho, Wo) = ctx.shape_io
        Sh, Sw = ctx.stride; Ph, Pw = ctx.padding; Dh, Dw = ctx.dilation
        BLOCK_M, BLOCK_N, BLOCK_K = ctx.blocks
        NUM_WARPS, NUM_STAGES = ctx.launch

        # всё в fp32
        gy32 = gy.float()
        w32  = w.float()
        x32  = x.float()

        M = N * Ho * Wo
        K = Cin * Kh * Kw

        # ---- dB ----
        gb = gy32.sum(dim=(0, 2, 3)) if ctx.has_bias else None

        # ---- dW = (img2col(x))^T @ dY ----
        cols = torch.empty((M, K), device=x.device, dtype=torch.float32)
        sN, sC, sH, sW = x32.stride()
        grid_i2c = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        img2col_kernel[grid_i2c](
            x32, cols,
            N, Cin, H, W_,
            Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
            Ho, Wo,
            sN, sC, sH, sW,
            K,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            CAST_FP16=False,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )
        dy_mat = gy32.permute(0, 2, 3, 1).contiguous().view(M, Cout)  # [M, Cout]
        cols_T = cols.t().contiguous()                                 # [K, M]

        dW_mat = triton_gemm(
            cols_T, dy_mat,
            use_fp16=False,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES
        )
        gw32 = dW_mat.view(Cin, Kh, Kw, Cout).permute(3, 0, 1, 2).contiguous()

        # ---- dX = col2img( dY @ W^T ) ----
        W_matT = w32.view(Cout, -1).contiguous()                       # [Cout, K]
        dcols  = triton_gemm(
            dy_mat, W_matT,
            use_fp16=False,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_K, BLOCK_K=Cout,  # (M,Cout)@(Cout,K)->(M,K)
            num_warps=NUM_WARPS, num_stages=NUM_STAGES
        )

        dx32 = torch.zeros((N, Cin, H, W_), device=x.device, dtype=torch.float32)
        sN, sC, sH, sW = dx32.stride()
        grid_c2i = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        col2img_kernel[grid_c2i](
            dcols, dx32,
            N, Cin, H, W_,
            Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
            Ho, Wo,
            sN, sC, sH, sW,
            K,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )

        dx = dx32.to(x.dtype, copy=False)
        gw = gw32.to(w.dtype, copy=False)
        gb = (gb.to(bias_stub.dtype, copy=False) if ctx.has_bias else None)
        # вернуть градиенты под все позиционные аргументы forward
        return dx, gw, gb, None, None, None, None, None, None, None, None, None, None