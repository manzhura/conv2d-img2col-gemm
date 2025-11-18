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
    def forward(ctx, x, w, bias, channel_mask, input_mask, stride, padding, dilation,
                BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
                NUM_WARPS=4, NUM_STAGES=2,
                ):
        assert x.is_cuda and w.is_cuda
        N, Cin, H, W_ = x.shape
        Cout, Cin_w, Kh, Kw = w.shape
        assert Cin == Cin_w
        Sh, Sw = stride; Ph, Pw = padding; Dh, Dw = dilation
        x_eff = x
        w_eff = w
        input_active_idx = None
        Cin_eff = Cin
        if input_mask is not None:
            mask_in = input_mask.to(device=x.device, dtype=torch.bool).view(-1)
            if mask_in.numel() != Cin:
                raise ValueError("input_channel_mask must match Cin")
            if not torch.all(mask_in):
                active_in = torch.nonzero(mask_in, as_tuple=False).flatten()
                if active_in.numel() == 0:
                    raise ValueError("input_channel_mask cannot prune all channels")
                input_active_idx = active_in
                x_eff = x.index_select(1, active_in)
                w_eff = w.index_select(1, active_in)
                Cin_eff = x_eff.shape[1]

        Ho, Wo = TritonConv2dFn._out_hw(H, W_, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw)
        M = N * Ho * Wo
        K = Cin_eff * Kh * Kw
        active_idx = None
        Cout_eff = Cout
        bias_eff = bias
        w_eff = w_eff
        if channel_mask is not None:
            mask = channel_mask.to(device=w.device, dtype=torch.bool).view(-1)
            if mask.numel() != Cout:
                raise ValueError("channel_mask must match Cout")
            if not torch.all(mask):
                active_idx = torch.nonzero(mask, as_tuple=False).flatten()
                if active_idx.numel() == 0:
                    raise ValueError("channel_mask cannot prune all channels")
        if active_idx is not None:
            w_eff = w.index_select(0, active_idx)
            bias_eff = bias.index_select(0, active_idx) if bias is not None else None
            Cout_eff = w_eff.shape[0]

        # ---- img2col -> cols[M,K] ----
        cols_dtype = torch.float16
        cols = torch.empty((M, K), device=x.device, dtype=cols_dtype)
        sN, sC, sH, sW = x_eff.stride()
        grid_i2c = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
        img2col_kernel[grid_i2c](
            x_eff, cols,
            N, Cin_eff, H, W_,
            Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
            Ho, Wo,
            sN, sC, sH, sW,
            K,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            CAST_FP16=(cols_dtype == torch.float16),
            num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )

        # ---- GEMM: [M,K] @ [K,Cout] -> [M,Cout] ----
        W_mat = w_eff.view(Cout_eff, -1).t().contiguous()
        y_col = triton_gemm(
            cols, W_mat,
            use_fp16=(cols_dtype == torch.float16),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=NUM_WARPS, num_stages=NUM_STAGES
        )  # fp32 output

        if bias_eff is not None:
            y_col.add_(bias_eff.float().view(1, -1))

        y = y_col.view(N, Ho, Wo, Cout_eff).permute(0, 3, 1, 2).contiguous()
        if active_idx is not None:
            y_full = torch.zeros((N, Cout, Ho, Wo), device=y.device, dtype=y.dtype)
            y_full[:, active_idx, :, :] = y
            y = y_full

        # save ctx
        ctx.save_for_backward(x, w, (bias if bias is not None else torch.tensor([], device=x.device, dtype=x.dtype)))
        ctx.has_bias = (bias is not None)
        ctx.stride   = stride
        ctx.padding  = padding
        ctx.dilation = dilation
        ctx.shape_io = (N, Cin_eff, H, W_, Cout, Kh, Kw, Ho, Wo)
        ctx.blocks   = (BLOCK_M, BLOCK_N, BLOCK_K)
        ctx.launch   = (NUM_WARPS, NUM_STAGES)
        ctx.active_idx = active_idx
        ctx.input_active_idx = input_active_idx
        ctx.Cout_eff = Cout_eff
        return y.to(x.dtype, copy=False)

