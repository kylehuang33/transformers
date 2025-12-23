"""AutoGrad version of pull/push/count/grad"""
import torch
from .bounds import BoundType
from .pushpull import (
    grid_pull, grid_pull_backward,
    grid_push, grid_push_backward,
    grid_count, grid_count_backward,
    grid_grad, grid_grad_backward
)


def make_list(x):
    if not isinstance(x, (list, tuple)):
        x = [x]
    return list(x)


def bound_convert(bound):

    if not isinstance(bound, (list, tuple)):
        bound = [bound]
    obound = []
    for b in bound:
        b = b.lower() if isinstance(b, str) else b
        if b in ('replicate', BoundType.replicate):
            obound.append('replicate')
        elif b in ('zeros', BoundType.zeros):
            obound.append('zeros')
        elif b in ('reflect', BoundType.reflect):
            obound.append('reflect')
        elif b in ('mirror', BoundType.mirror):
            obound.append('mirror')
        elif isinstance(b, int):
            obound.append(b)
        else:
            raise ValueError(f'Unknown boundary condition {b}')
    obound = list(map(lambda b: getattr(BoundType, b) if isinstance(b, str)
                      else BoundType(b), obound))
    obound = [b.value for b in obound]

    return obound


class GridPull(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        bound = bound_convert(make_list(bound))
        interpolation = make_list(interpolation)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        output = grid_pull(input, grid, *opt)

        ctx.opt = opt
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grads = grid_pull_backward(grad, *var, *opt)
        grad_input, grad_grid = grads
        return grad_input, grad_grid, None, None, None


class GridPush(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid, shape, interpolation, bound, extrapolate):

        bound = bound_convert(make_list(bound))
        interpolation = make_list(interpolation)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        output = grid_push(input, grid, shape, *opt)

        ctx.opt = opt
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grads = grid_push_backward(grad, *var, *opt)
        grad_input, grad_grid = grads
        return grad_input, grad_grid, None, None, None, None


class GridCount(torch.autograd.Function):

    @staticmethod
    def forward(ctx, grid, shape, interpolation, bound, extrapolate):

        bound = bound_convert(make_list(bound))
        interpolation = make_list(interpolation)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        output = grid_count(grid, shape, *opt)

        ctx.opt = opt
        ctx.save_for_backward(grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grad_grid = None
        if ctx.needs_input_grad[0]:
            grad_grid = grid_count_backward(grad, *var, *opt)
        return grad_grid, None, None, None, None


class GridGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, grid, interpolation, bound, extrapolate):

        bound = bound_convert(make_list(bound))
        interpolation = make_list(interpolation)
        extrapolate = int(extrapolate)
        opt = (bound, interpolation, extrapolate)

        output = grid_grad(input, grid, *opt)

        ctx.opt = opt
        ctx.save_for_backward(input, grid)

        return output

    @staticmethod
    def backward(ctx, grad):
        var = ctx.saved_tensors
        opt = ctx.opt
        grad_input = grad_grid = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grads = grid_grad_backward(grad, *var, *opt)
            grad_input, grad_grid = grads
        return grad_input, grad_grid, None, None, None