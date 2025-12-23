import torch
from enum import Enum
from typing import Optional
Tensor = torch.Tensor


class BoundType(Enum):
    zeros = 0
    replicate = 1
    mirror = 2
    reflect = 3

class ExtrapolateType(Enum):
    no = 0     # threshold: (0, n-1)
    yes = 1
    hist = 2   # threshold: (-0.5, n-0.5)


@torch.jit.script
class Bound:

    def __init__(self, bound_type: int = 3):
        self.type = bound_type

    def index(self, i, n: int):
        if self.type in (0, 1):  # zero / replicate
            return i.clamp(min=0, max=n-1)
        elif self.type == 2:  # mirror
            if n == 1:
                return torch.zeros(i.shape, dtype=i.dtype, device=i.device)
            else:
                n2 = (n - 1) * 2
                i = i.abs().remainder(n2)
                i = torch.where(i >= n, -i + n2, i)
                return i
        elif self.type == 3: # reflect
            n2 = n * 2
            i = torch.where(i < 0, (-i-1).remainder(n2).neg().add(n2 - 1),
                            i.remainder(n2))
            i = torch.where(i >= n, -i + (n2 - 1), i)
            return i
        else:
            return i

    def transform(self, i, n: int) -> Optional[Tensor]:
        if self.type == 0:
            one = torch.ones([1], dtype=torch.int8, device=i.device)
            zero = torch.zeros([1], dtype=torch.int8, device=i.device)
            outbounds = ((i < 0) | (i >= n))
            x = torch.where(outbounds, zero, one)
            return x
        else:
            return None
