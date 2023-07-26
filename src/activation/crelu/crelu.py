import torch
from torch import nn, Tensor
import torch.nn.functional as F


class CReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input:Tensor, dim:int=-1, negative_slope:float=0.01) -> Tensor:
        ctx.dim = dim
        ctx.save_for_backward(input)
        ctx.negative_slope = negative_slope

        return F.leaky_relu(torch.cat((input, torch.neg(input)), dim), negative_slope)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        input, = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()

        half_size = grad_input.size(dim) // 2
        pos_part = grad_input.narrow(dim, 0, half_size)
        neg_part = grad_input.narrow(dim, half_size, half_size)

        pos_part[input < 0] = ctx.negative_slope
        neg_part[input >= 0] = ctx.negative_slope

        grad_input = pos_part - neg_part

        return grad_input, None, None


crelu_cuda = CReLUFunction.apply

def crelu_pt(x:Tensor, dim:int=-1, negative_slope:float=0.01) -> Tensor:
    return F.leaky_relu(torch.cat((x, torch.neg(x)), dim), negative_slope)


def test_backward_creluFunction(num_loop:int=100):
    from random import random, seed, randint

    seed(10)
    g=torch.Generator().manual_seed(10)

    def compare_grad(in_ft:int,out_ft:int, negative_slope:float
                     ) -> tuple[torch.bool, torch.bool]:
        mlp_cuda = nn.Linear(in_ft, out_ft)
        mlp_pt = nn.Linear(in_ft, out_ft)

        # init 2 difference layer
        # with torch.no_grad():
        weight = torch.randn((out_ft, in_ft), requires_grad=True)
        bias = torch.randn((1, ), requires_grad=True, generator=g)
        mlp_cuda.weight = nn.Parameter(weight)
        mlp_cuda.bias = nn.Parameter(bias)

        mlp_pt.weight = nn.Parameter(weight)
        mlp_pt.bias = nn.Parameter(bias)

        x = torch.randn((in_ft, ), generator=g)
        # x = torch.tensor([1.5, 0, -0.6], requires_grad=False, dtype=torch.float32)
        negative_slope=random()
        y_cuda = torch.sum(crelu_cuda(mlp_cuda(x), -1, negative_slope))
        y_cuda.backward()
        y_pt   = torch.sum(crelu_pt  (mlp_pt(x), -1, negative_slope))
        y_pt.backward()

        sub_l = y_cuda.data - y_pt.data
        #print('sub_l = ', sub_l)
        sub = mlp_cuda.weight.grad.data - mlp_pt.weight.grad.data
        # print(mlp_cuda.weight.grad.data)
        # print(mlp_pt.weight.grad.data)
        return sub.sum() == 0, sub_l.sum() == 0


    for i in range(num_loop):
        rs = compare_grad(
            in_ft = randint(2, 3),
            out_ft = randint(2, 3),
            negative_slope = random()
        )
        print(i, ' -> ', rs)

test_backward_creluFunction(0)
# del test_backward_creluFunction
