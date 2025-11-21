import torch
import triton

from conv_gemm.triton_kernels.fp16.img2col_kernel import img2col_kernel
from conv_gemm.triton_kernels.fp16.col2img_kernel import col2img_kernel
from conv_gemm.triton_kernels.fp16.gemm_kernel import triton_gemm

from conv_gemm.configs.kernel_config import FP16_I2C_CFG, FP16_GEMM_CFG,FP32_C2I_CFG


class TritonConv2dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, w, bias,
        channel_mask, input_mask, grad_channel_mask,
        grad_input_mask, stride, padding, dilation,
    ):
        """
        Forward-проход свёртки на Triton с динамической спарсификацией.
        Возможности:
        • FP16 forward: img2col → GEMM → col2img.
        • Учитывает sparsity по входным и выходным каналам:
            - input_mask уменьшает Cin_eff и K = Cin_eff * Kh * Kw.
            - channel_mask уменьшает Cout_eff.
        • GEMM реально работает на сжатых матрицах [M, K_eff] @ [K_eff, Cout_eff].
        • Маски grad_* только сохраняются в ctx — используются в backward().
        • Реконструирует полный выходной тензор, заполняя отрубленные каналы нулями.
        Возвращает:
        • y — результат свёртки с учётом sparsity (тип как у входа).
        """
        assert x.is_cuda and w.is_cuda

        N, Cin, H, W_ = x.shape
        Cout, Cin_w, Kh, Kw = w.shape
        assert Cin == Cin_w

        Sh, Sw = stride
        Ph, Pw = padding
        Dh, Dw = dilation

        # input mask (forward)
        input_mask_bool = torch.ones(Cin, device=x.device, dtype=torch.bool)
        if input_mask is not None:
            mask_in = input_mask.to(device=x.device, dtype=torch.bool).view(-1)
            if mask_in.numel() != Cin:
                raise ValueError("input_channel_mask must match Cin")
            input_mask_bool = mask_in
        if not torch.any(input_mask_bool):
            raise ValueError("input_channel_mask cannot prune all channels")

        input_active_idx = None
        x_eff = x
        w_eff = w
        Cin_eff = int(input_mask_bool.sum().item())
        if not torch.all(input_mask_bool):
            input_active_idx = torch.nonzero(input_mask_bool, as_tuple=False).flatten()
            if input_active_idx.numel() == 0:
                raise ValueError("input_channel_mask cannot prune all channels")
            x_eff = x.index_select(1, input_active_idx)
            w_eff = w.index_select(1, input_active_idx)

        # channel mask (forward)
        channel_mask_bool = torch.ones(Cout, device=w.device, dtype=torch.bool)
        if channel_mask is not None:
            mask = channel_mask.to(device=w.device, dtype=torch.bool).view(-1)
            if mask.numel() != Cout:
                raise ValueError("channel_mask must match Cout")
            channel_mask_bool = mask
        if not torch.any(channel_mask_bool):
            raise ValueError("channel_mask cannot prune all channels")

        active_idx = None
        Cout_eff = int(channel_mask_bool.sum().item())
        bias_eff = bias
        if not torch.all(channel_mask_bool):
            active_idx = torch.nonzero(channel_mask_bool, as_tuple=False).flatten()
            if active_idx.numel() == 0:
                raise ValueError("channel_mask cannot prune all channels")
            w_eff = w_eff.index_select(0, active_idx)
            bias_eff = bias.index_select(0, active_idx) if bias is not None else None

        # gradient masks
        grad_input_mask_bool = input_mask_bool
        grad_input_override = False
        grad_input_active_idx = None
        if grad_input_mask is not None:
            mask_grad_in = grad_input_mask.to(device=x.device, dtype=torch.bool).view(-1)
            if mask_grad_in.numel() != Cin:
                raise ValueError("grad_input_mask must match Cin")
            grad_input_mask_bool = mask_grad_in & input_mask_bool
            if not torch.any(grad_input_mask_bool):
                raise ValueError("grad_input_mask cannot prune all channels")
            if not torch.equal(grad_input_mask_bool, input_mask_bool):
                grad_input_override = True
                grad_input_active_idx = torch.nonzero(grad_input_mask_bool, as_tuple=False).flatten()
        Cin_grad_eff = int(grad_input_mask_bool.sum().item())
        if not grad_input_override:
            grad_input_active_idx = input_active_idx

        grad_channel_mask_bool = channel_mask_bool
        grad_channel_override = False
        grad_active_idx = None
        if grad_channel_mask is not None:
            mask_grad = grad_channel_mask.to(device=w.device, dtype=torch.bool).view(-1)
            if mask_grad.numel() != Cout:
                raise ValueError("grad_channel_mask must match Cout")
            grad_channel_mask_bool = mask_grad & channel_mask_bool
            if not torch.any(grad_channel_mask_bool):
                raise ValueError("grad_channel_mask cannot prune all channels")
            if not torch.equal(grad_channel_mask_bool, channel_mask_bool):
                grad_channel_override = True
                grad_active_idx = torch.nonzero(grad_channel_mask_bool, as_tuple=False).flatten()
        Cout_grad_eff = int(grad_channel_mask_bool.sum().item())
        if not grad_channel_override:
            grad_active_idx = active_idx

        Ho, Wo = TritonConv2dFn._out_hw(H, W_, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw)
        M = N * Ho * Wo
        K = Cin_eff * Kh * Kw

        # img2col -> cols[M,K]
        cols_dtype = torch.float16
        cols = torch.empty((M, K), device=x.device, dtype=cols_dtype)
        sN, sC, sH, sW = x_eff.stride()
        grid_i2c = (triton.cdiv(M, FP16_I2C_CFG.BLOCK_M), triton.cdiv(K, FP16_I2C_CFG.BLOCK_K))
        img2col_kernel[grid_i2c](
            x_eff, cols,
            N, Cin_eff, H, W_,
            Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
            Ho, Wo,
            sN, sC, sH, sW,
            K,
            BLOCK_M=FP16_I2C_CFG.BLOCK_M, BLOCK_K=FP16_I2C_CFG.BLOCK_K,
            CAST_FP16=(cols_dtype == torch.float16), #  fp16 forward
            num_warps=FP16_I2C_CFG.NUM_WARPS, num_stages=FP16_I2C_CFG.NUM_STAGES,
        )

        # GEMM [M,K] @ [K,Cout] -> [M,Cout]
        W_mat = w_eff.view(Cout_eff, -1).t().contiguous()
        y_col = triton_gemm(
            cols, W_mat,
            use_fp16=(cols_dtype == torch.float16),
            BLOCK_M=FP16_GEMM_CFG.BLOCK_M, BLOCK_N=FP16_GEMM_CFG.BLOCK_N, BLOCK_K=FP16_GEMM_CFG.BLOCK_K,
            num_warps=FP16_GEMM_CFG.NUM_WARPS, num_stages=FP16_GEMM_CFG.NUM_STAGES
        )

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
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.shape_io = (N, Cin, H, W_, Cout, Kh, Kw, Ho, Wo)
        ctx.active_idx = active_idx
        ctx.input_active_idx = input_active_idx
        ctx.grad_active_idx = grad_active_idx
        ctx.grad_input_active_idx = grad_input_active_idx
        ctx.grad_channel_override = grad_channel_override
        ctx.grad_input_override = grad_input_override
        ctx.Cout_eff = Cout_eff
        ctx.Cin_eff = Cin_eff
        ctx.Cout_grad_eff = Cout_grad_eff
        ctx.Cin_grad_eff = Cin_grad_eff
        return y.to(x.dtype, copy=False)

    @staticmethod
    def backward(ctx, gy):
        x, w, bias_stub = ctx.saved_tensors
        (N, Cin_full, H, W_, Cout_full, Kh, Kw, Ho, Wo) = ctx.shape_io
        Sh, Sw = ctx.stride
        Ph, Pw = ctx.padding
        Dh, Dw = ctx.dilation
        active_idx = ctx.active_idx
        input_active_idx = ctx.input_active_idx
        grad_channel_override = ctx.grad_channel_override
        grad_input_override = ctx.grad_input_override

        if grad_channel_override:
            out_idx = ctx.grad_active_idx
            Cout_work = ctx.Cout_grad_eff
        elif active_idx is not None:
            out_idx = active_idx
            Cout_work = ctx.Cout_eff
        else:
            out_idx = None
            Cout_work = Cout_full

        if grad_input_override:
            in_idx = ctx.grad_input_active_idx
            Cin_work = ctx.Cin_grad_eff
        elif input_active_idx is not None:
            in_idx = input_active_idx
            Cin_work = ctx.Cin_eff
        else:
            in_idx = None
            Cin_work = Cin_full

        if Cout_work <= 0 or Cin_work <= 0:
            raise ValueError("Sparsity masks cannot prune all channels in backward pass")

        gypsum = gy.float()
        if out_idx is not None:
            gy_eff = gypsum.index_select(1, out_idx)
        else:
            gy_eff = gypsum

        if ctx.has_bias:
            gb_eff = gy_eff.sum(dim=(0, 2, 3))
            if out_idx is not None:
                gb_full = torch.zeros(Cout_full, device=gy_eff.device, dtype=torch.float32)
                gb_full.index_copy_(0, out_idx, gb_eff)
            else:
                gb_full = gb_eff
        else:
            gb_full = None

        if in_idx is not None:
            x_eff32 = x.index_select(1, in_idx).float()
        else:
            x_eff32 = x.float()

        w32 = w.float()
        if in_idx is not None:
            w32 = w32.index_select(1, in_idx)
        if out_idx is not None:
            w32 = w32.index_select(0, out_idx)

        M = N * Ho * Wo
        K_grad = Cin_work * Kh * Kw

        cols = torch.empty((M, K_grad), device=x.device, dtype=torch.float32)
        sN, sC, sH, sW = x_eff32.stride()
        grid_i2c = (triton.cdiv(M, FP16_I2C_CFG.BLOCK_M), triton.cdiv(K_grad, FP16_I2C_CFG.BLOCK_K))
        img2col_kernel[grid_i2c](
            x_eff32, cols,
            N, Cin_work, H, W_,
            Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
            Ho, Wo,
            sN, sC, sH, sW,
            K_grad,
            BLOCK_M=FP16_I2C_CFG.BLOCK_M, BLOCK_K=FP16_I2C_CFG.BLOCK_K,
            CAST_FP16=False, #  fp16 backward
            num_warps=FP16_I2C_CFG.NUM_WARPS, num_stages=FP16_I2C_CFG.NUM_STAGES,
        )

        dy_mat = gy_eff.permute(0, 2, 3, 1).contiguous().view(M, Cout_work)
        cols_T = cols.t().contiguous()

        dW_mat = triton_gemm(
            cols_T, dy_mat,
            use_fp16=False,
            BLOCK_M=FP16_GEMM_CFG.BLOCK_M, BLOCK_N=FP16_GEMM_CFG.BLOCK_N, BLOCK_K=FP16_GEMM_CFG.BLOCK_K,
            num_warps=FP16_GEMM_CFG.NUM_WARPS, num_stages=FP16_GEMM_CFG.NUM_STAGES,
        )
        gw_grad32 = dW_mat.view(Cin_work, Kh, Kw, Cout_work).permute(3, 0, 1, 2).contiguous()

        if in_idx is not None:
            gw_input = torch.zeros((Cout_work, Cin_full, Kh, Kw), device=w.device, dtype=torch.float32)
            gw_input.index_copy_(1, in_idx, gw_grad32)
        else:
            gw_input = gw_grad32
        if out_idx is not None:
            gw_full = torch.zeros((Cout_full, Cin_full, Kh, Kw), device=w.device, dtype=torch.float32)
            gw_full.index_copy_(0, out_idx, gw_input)
        else:
            gw_full = gw_input

        W_matT = w32.view(Cout_work, -1).contiguous()
        dcols = triton_gemm(
            dy_mat, W_matT,
            use_fp16=False,
            BLOCK_M=FP16_GEMM_CFG.BLOCK_M, BLOCK_N=FP16_GEMM_CFG.BLOCK_K, BLOCK_K=FP16_GEMM_CFG.BLOCK_K,
            num_warps=FP16_GEMM_CFG.NUM_WARPS, num_stages=FP16_GEMM_CFG.NUM_STAGES,
        )

        dx_eff32 = torch.zeros((N, Cin_work, H, W_), device=x.device, dtype=torch.float32)
        sN_dx, sC_dx, sH_dx, sW_dx = dx_eff32.stride()
        grid_c2i = (triton.cdiv(M, FP32_C2I_CFG.BLOCK_M), triton.cdiv(K_grad, FP32_C2I_CFG.BLOCK_K))
        col2img_kernel[grid_c2i](
            dcols, dx_eff32,
            N, Cin_work, H, W_,
            Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw,
            Ho, Wo,
            sN_dx, sC_dx, sH_dx, sW_dx,
            K_grad,
            BLOCK_M=FP32_C2I_CFG.BLOCK_M, BLOCK_K=FP32_C2I_CFG.BLOCK_K,
            num_warps=FP32_C2I_CFG.NUM_WARPS, num_stages=FP32_C2I_CFG.NUM_STAGES,
        ) # fp32 для производной
        if in_idx is not None:
            dx32 = torch.zeros((N, Cin_full, H, W_), device=x.device, dtype=torch.float32)
            dx32.index_copy_(1, in_idx, dx_eff32)
        else:
            dx32 = dx_eff32

        dx = dx32.to(x.dtype, copy=False)
        gw = gw_full.to(w.dtype, copy=False)
        gb = (gb_full.to(bias_stub.dtype, copy=False) if ctx.has_bias else None)

        return dx, gw, gb, None, None, None, None, None, None, None, None, None, None, None, None

    @staticmethod
    def _out_hw(H, W, Kh, Kw, Sh, Sw, Ph, Pw, Dh, Dw):
        Ho = (H + 2 * Ph - Dh * (Kh - 1) - 1) // Sh + 1
        Wo = (W + 2 * Pw - Dw * (Kw - 1) - 1) // Sw + 1
        return Ho, Wo
