"""High level interpolation API"""
__all__ = ['grid_pull', 'grid_push', 'grid_count', 'grid_grad']

import torch
from .utils import expanded_shape
from .autograd import (GridPull, GridPush, GridCount, GridGrad)

_doc_order = """
    The order of the spline interpolation, default is 1 for linear. 
    The order has to be in the range 0-7.
    - 0 'nearest' (default)
    - 1 'linear'
    - 2 'quadratic'
    - 3 'cubic'
    - 4 'fourth'
    - etc.
"""

_doc_mode = """
    The mode parameter determines how the input array is extended beyond its boundaries. 
    Default is 'zeros'. Behavior for each valid value is as follows.
    - 'reflect'    (c b a | a b c d | d c b)
    - 'mirror'     (d c b | a b c d | c b a)
    - 'zeros'      (0 0 0 | a b c d | 0 0 0)
    - 'replicate'  (a a a | a b c d | d d d)
"""

def _preproc(grid, inp=None, mode=None):

    dim = grid.shape[-1]
    if inp is None:
        spatial = grid.shape[-dim-1:-1]
        batch = grid.shape[:-dim-1]
        grid = grid.reshape([-1, *spatial, dim])
        info = dict(batch=batch, channel=[1] if batch else [], dim=dim)
        return grid, info

    grid_spatial = grid.shape[-dim-1:-1]
    grid_batch = grid.shape[:-dim-1]
    input_spatial = inp.shape[-dim:]
    channel = 0 if inp.dim() == dim else inp.shape[-dim-1]
    input_batch = inp.shape[:-dim-1]

    if mode == 'push':
        grid_spatial = input_spatial = expanded_shape(grid_spatial, input_spatial)

    # broadcast and reshape
    batch = expanded_shape(grid_batch, input_batch)
    grid = grid.expand([*batch, *grid_spatial, dim])
    grid = grid.reshape([-1, *grid_spatial, dim])
    inp = inp.expand([*batch, channel or 1, *input_spatial])
    inp = inp.reshape([-1, channel or 1, *input_spatial])

    out_channel = [channel] if channel else ([1] if batch else [])
    info = dict(batch=batch, channel=out_channel, dim=dim)
    return grid, inp, info


def _postproc(out, shape_info, mode):

    dim = shape_info['dim']
    if mode != 'grad':
        spatial = out.shape[-dim:]
        feat = []
    else:
        spatial = out.shape[-dim-1:-1]
        feat = [out.shape[-1]]
    batch = shape_info['batch']
    channel = shape_info['channel']

    out = out.reshape([*batch, *channel, *spatial, *feat])
    return out


def grid_pull(input, grid, order=1, mode='replicate',
              extrapolate=False):
    """Sample an image with respect to a deformation field.

    Notes
    -----
    {order}

    {mode}

    If the input dtype is not a floating point type, the input image is 
    assumed to contain labels. Then, unique labels are extracted 
    and resampled individually, making them soft labels. Finally, 
    the label map is reconstructed from the individual soft labels by 
    assigning the label with maximum soft value.

    Parameters
    ----------
    input : (..., [channel], *inshape) tensor
        Input image.
    grid : (..., *outshape, dim) tensor
        Transformation field.
    order : int or sequence[int], default=1
        Interpolation order.
    mode : BoundType or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : (..., [channel], *outshape) tensor
        Deformed image.

    """

    grid, input, shape_info = _preproc(grid, input)
    batch, channel = input.shape[:2]

    if not input.dtype.is_floating_point:
        # label map -> specific processing
        out = input.new_zeros([batch, channel, *grid.shape[1:-1]])
        pmax = grid.new_zeros([batch, channel, *grid.shape[1:-1]])
        for label in input.unique():
            soft = (input == label).to(grid.dtype)
            soft = GridPull.apply(soft, grid, order, mode, extrapolate)
            out[soft > pmax] = label
            pmax = torch.max(pmax, soft)
    else:
        out = GridPull.apply(input, grid, order, mode, extrapolate)

    return _postproc(out, shape_info, mode='pull')


def grid_push(input, grid, shape=None, order=1, mode='replicate',
              extrapolate=False):
    """Splat an image with respect to a deformation field (pull adjoint).

    Notes
    -----
    {order}

    {mode}

    Parameters
    ----------
    input : (..., [channel], *inshape) tensor
        Input image.
    grid : (..., *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    order : int or sequence[int], default=1
        Interpolation order.
    mode : BoundType, or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : (..., [channel], *shape) tensor
        Spatted image.

    """

    grid, input, shape_info = _preproc(grid, input, mode='push')

    if shape is None:
        shape = tuple(input.shape[2:])

    out = GridPush.apply(input, grid, shape, order, mode, extrapolate)

    out = _postproc(out, shape_info, mode='push')

    return out


def grid_count(grid, shape=None, order=1, mode='replicate',
               extrapolate=False):
    """Splatting weights with respect to a deformation field (pull adjoint).

    Notes
    -----
    {order}

    {mode}

    Parameters
    ----------
    grid : (..., *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    order : int or sequence[int], default=1
        Interpolation order.
    mode : BoundType, or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : (..., [1], *shape) tensor
        Splatted weights.

    """

    grid, shape_info = _preproc(grid)
    out = GridCount.apply(grid, shape, order, mode, extrapolate)
    return _postproc(out, shape_info, mode='count')


def grid_grad(input, grid, order=1, mode='replicate',
              extrapolate=False):
    """Sample spatial gradients of an image with respect to a deformation field.
    
    Notes
    -----
    {order}

    {mode}

    Parameters
    ----------
    input : (..., [channel], *inshape) tensor
        Input image.
    grid : (..., *inshape, dim) tensor
        Transformation field.
    shape : sequence[int], default=inshape
        Output shape
    order : int or sequence[int], default=1
        Interpolation order.
    mode : BoundType, or sequence[BoundType], default='zero'
        Boundary conditions.
    extrapolate : bool or int, default=True
        Extrapolate out-of-bound data.

    Returns
    -------
    output : (..., [channel], *shape, dim) tensor
        Sampled gradients.

    """

    grid, input, shape_info = _preproc(grid, input)
    out = GridGrad.apply(input, grid, order, mode, extrapolate)
    return _postproc(out, shape_info, mode='grad')



grid_pull.__doc__ = grid_pull.__doc__.format(
    order=_doc_order, mode=_doc_mode)
grid_push.__doc__ = grid_push.__doc__.format(
    order=_doc_order, mode=_doc_mode)
grid_count.__doc__ = grid_count.__doc__.format(
    order=_doc_order, mode=_doc_mode)
grid_grad.__doc__ = grid_grad.__doc__.format(
    order=_doc_order, mode=_doc_mode)