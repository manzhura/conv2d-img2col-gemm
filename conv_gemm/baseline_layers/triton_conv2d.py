# === TritonConv2d — nn.Module над твоим TritonConv2dFn ===
import math
import torch, triton
from conv_gemm.baseline_operators.triton_conv2d_fp16_fn import TritonConv2dFn




class TritonConv2d(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size,
        stride=1, padding=0, dilation=1, bias=True,
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
        NUM_WARPS=4, NUM_STAGES=2,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        if isinstance(stride, int):   stride = (stride, stride)
        if isinstance(padding, int):  padding = (padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation)

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = (kh, kw)
        self.stride       = stride
        self.padding      = padding
        self.dilation     = dilation

        # --- weights INIT (СРАЗУ FP16) ---
        w = torch.empty(out_channels, in_channels, kh, kw)
        torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
        self.weight = torch.nn.Parameter(w.half())

        # --- bias ---
        if bias:
            fan_in = in_channels * kh * kw
            bound = 1 / math.sqrt(fan_in)
            b = torch.empty(out_channels)
            torch.nn.init.uniform_(b, -bound, bound)
            self.bias = torch.nn.Parameter(b.float())  # FP32 (лучше для накопления)
        else:
            self.bias = None


        # Тюнинг кернелов
        self.BLOCK_M   = BLOCK_M
        self.BLOCK_N   = BLOCK_N
        self.BLOCK_K   = BLOCK_K
        self.NUM_WARPS = NUM_WARPS
        self.NUM_STAGES = NUM_STAGES

        self.register_buffer(
            "channel_mask",
            torch.ones(out_channels, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "input_channel_mask",
            torch.ones(in_channels, dtype=torch.bool),
            persistent=True,
        )
        self.block_size = None


    def forward(self, x):
        assert x.is_cuda and self.weight.is_cuda
        x_in = x.half()
        w_in = self.weight.half()

        y = TritonConv2dFn.apply(
            x_in, w_in, self.bias,
            self._resolved_channel_mask(),
            self._resolved_input_mask(),
            self.stride, self.padding, self.dilation,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K,
            self.NUM_WARPS, self.NUM_STAGES,
        )
        return y # мб халф

    # ===== Sparsity helpers =====
    def set_channel_mask(self, mask: torch.Tensor | None):
        if mask is None:
            self.channel_mask = torch.ones(
                self.out_channels, device=self.weight.device, dtype=torch.bool
            )
            return
        mask = mask.to(device=self.weight.device, dtype=torch.bool).view(-1)
        if mask.numel() != self.out_channels:
            raise ValueError("channel_mask must have length == out_channels")
        self.channel_mask = mask
        self.block_size = None
        self.input_channel_mask.fill_(True)

    def clear_sparsity(self):
        self.channel_mask.fill_(True)
        self.input_channel_mask.fill_(True)
        self.block_size = None

    def set_channel_sparsity(self, keep_ratio: float):
        keep_ratio = float(keep_ratio)
        if not (0 < keep_ratio <= 1):
            raise ValueError("keep_ratio must be in (0,1]")
        keep = max(1, int(round(self.out_channels * keep_ratio)))
        with torch.no_grad():
            scores = self.weight.detach().abs().sum(dim=(1, 2, 3))
            topk = torch.topk(scores, k=keep, largest=True).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[topk] = True
        self.channel_mask = mask
        self.block_size = None

    def set_block_sparsity(self, keep_ratio: float, block_size: int = 4):
        keep_ratio = float(keep_ratio)
        if not (0 < keep_ratio <= 1):
            raise ValueError("keep_ratio must be in (0,1]")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        num_blocks = (self.out_channels + block_size - 1) // block_size
        keep_blocks = max(1, int(round(num_blocks * keep_ratio)))
        with torch.no_grad():
            scores = self.weight.detach().abs().sum(dim=(1, 2, 3))
            pad = num_blocks * block_size - scores.numel()
            if pad > 0:
                scores = torch.cat([scores, torch.zeros(pad, device=scores.device)], dim=0)
            block_scores = scores.view(num_blocks, block_size).sum(dim=1)
            topk = torch.topk(block_scores, k=keep_blocks, largest=True).indices
            mask = torch.zeros(num_blocks, block_size, device=scores.device, dtype=torch.bool)
            mask[topk] = True
            mask = mask.view(-1)[:self.out_channels]
        self.channel_mask = mask
        self.block_size = block_size
        self.input_channel_mask.fill_(True)

    def set_input_channel_mask(self, mask: torch.Tensor | None):
        if mask is None:
            self.input_channel_mask = torch.ones(
                self.in_channels, device=self.weight.device, dtype=torch.bool
            )
            return
        mask = mask.to(device=self.weight.device, dtype=torch.bool).view(-1)
        if mask.numel() != self.in_channels:
            raise ValueError("input_channel_mask must have length == in_channels")
        self.input_channel_mask = mask

    def set_input_channel_sparsity(self, keep_ratio: float):
        keep_ratio = float(keep_ratio)
        if not (0 < keep_ratio <= 1):
            raise ValueError("keep_ratio must be in (0,1]")
        keep = max(1, int(round(self.in_channels * keep_ratio)))
        with torch.no_grad():
            scores = self.weight.detach().abs().sum(dim=(0, 2, 3))
            topk = torch.topk(scores, k=keep, largest=True).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask[topk] = True
        self.input_channel_mask = mask

    def clear_input_sparsity(self):
        self.input_channel_mask.fill_(True)

    def _resolved_channel_mask(self):
        mask = getattr(self, "channel_mask", None)
        if mask is None:
            return None
        mask = mask.to(device=self.weight.device, dtype=torch.bool).view(-1)
        if mask.numel() != self.out_channels:
            raise ValueError("channel_mask must match out_channels")
        if torch.all(mask):
            return None
        return mask

    def _resolved_input_mask(self):
        mask = getattr(self, "input_channel_mask", None)
        if mask is None:
            return None
        mask = mask.to(device=self.weight.device, dtype=torch.bool).view(-1)
        if mask.numel() != self.in_channels:
            raise ValueError("input_channel_mask must match in_channels")
        if torch.all(mask):
            return None
        return mask


