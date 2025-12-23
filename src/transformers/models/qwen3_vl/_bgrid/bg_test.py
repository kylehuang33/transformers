import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as nnf
from _bgrid.bgrid_layer import bgridSplatting2DTo3D


if __name__ == "__main__":
    # simple test
    n, c, h, w = 4, 4096, 16, 16
    img_size = (h,w)
    x = torch.randn(n,c,h,w).float()
    gm = torch.randn(n,1,448,448).float()
    model = bgridSplatting2DTo3D(img_size, s_s=0.5, s_r=8, order=1, mode='replicate')
    # with torch.no_grad():
    bg, on = model(x, gm)
    print(bg.shape, on.shape)
    
    print((on>0.001).sum())
