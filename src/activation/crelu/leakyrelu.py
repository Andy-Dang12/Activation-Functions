from ast import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class LeakyreluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:Tensor, negative_slope:float=0.01) -> Tensor:
        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope
        return F.leaky_relu(input, negative_slope)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> 'Tuple[Tensor, None]':
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # The Leaky ReLU function does not have a derivative at x=0
        # follow backward of F.leaky_relu,
        # I choose `backward` at x=0 is `negative_slope`
        mask = input>0
        grad_input[~mask] = ctx.negative_slope
        grad_input[mask] = 1.0

        return grad_input, None


leaky1 = LeakyreluFunction.apply
leaky2 = F.leaky_relu


def compare_backward_leakyrelu():
    from random import randint, seed
    seed(10)

    for _ in range(1000):
        x1 = torch.randn((randint(1, 1000), ), requires_grad=True, dtype=torch.float32)
        x2 = torch.tensor(x1, requires_grad=True, dtype=torch.float32)

        alpha1 = 0.0
        alpha2 = 0.1
        z1 = leaky1(x1, alpha1)
        z2 = leaky2(x2, alpha2)
        tong1 = torch.sum(z1)
        tong1.backward()

        tong2 =torch.sum(z2)
        tong2.backward()

        sub = x1.grad.data - x2.grad.data
        print(sub.sum())

compare_backward_leakyrelu()
