__all__ = ['Mish','MishFunction','mish_forward','mish_backward']

import torch # Must import torch before C extension
from ._C import mish_forward, mish_backward

class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        ctx.save_for_backward(inp)
        return mish_forward(inp)

    @staticmethod
    def backward(ctx, grad_out):
        inp, = ctx.saved_tensors
        if not ctx.needs_input_grad[0]: return (None,)
        return mish_backward(inp, grad_out)


class Mish(torch.nn.Module):
    def forward(self, inp): return MishFunction.apply(inp)
