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
        if isinstance(kernel_size, int): kh = kw = kernel_size
        else: kh, kw = kernel_size
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


    def forward(self, x):
        assert x.is_cuda and self.weight.is_cuda
        x_in = x.half()
        w_in = self.weight.half()

        y = TritonConv2dFn.apply(
            x_in, w_in, self.bias,
            self.stride, self.padding, self.dilation,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K,
            self.NUM_WARPS, self.NUM_STAGES,
        )
        return y # мб халф