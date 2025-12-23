import math
import torch
import numbers
import numpy as np
import torch.nn as nn
import torch.nn.functional as nnf
from .dags import grid_push, grid_pull

class bgridSlicing4DTo3D(nn.Module):

    def __init__(self, img_size, order=1, mode='replicate'):

        super(bgridSlicing4DTo3D, self).__init__()

        self.img_size = img_size
        self.order = order
        self.mode = mode
        h,w,d = img_size
        ohg, owg, ozg = torch.meshgrid([torch.arange(0, h),torch.arange(0, w),torch.arange(0, d)])
        hg = ohg.type_as(ozg).to(ozg.device) / (h-1)
        wg = owg.type_as(ozg).to(ozg.device) / (w-1)
        dg = ozg.type_as(ozg).to(ozg.device) / (d-1)
        self.register_buffer('hg', hg)
        self.register_buffer('wg', wg)
        self.register_buffer('dg', dg)

    def forward(self, bg, gm):

        '''
        bg: (n,c,h,w,d,up)
        gm: (n,1,gh,gw,gd)
        '''
        n,c,h,w,d,up = bg.shape
        _,_,gh,gw,gd = gm.shape

        gm = gm * (up-1)
        hg = self.hg.view(1,gh,gw,gd).repeat(n,1,1,1)*(h-1)
        wg = self.wg.view(1,gh,gw,gd).repeat(n,1,1,1)*(w-1)
        dg = self.dg.view(1,gh,gw,gd).repeat(n,1,1,1)*(d-1)

        gm = gm[:,0] # n,gh,gw,gd
        gm = torch.stack([hg, wg, dg, gm], dim=-1).unsqueeze(1) # n,1,gh,gw,gd,4
        bg = grid_pull(bg, gm, order=self.order,mode=self.mode) # n,c,1,h,w,d
        bg = bg.squeeze(2) # n,c,h,w,d

        return bg

class bgridSplatting2DTo3D(nn.Module):
    '''
    mode 1: linear interpolation
    mode 2: quadratic interpolation
    mode 3: cubic interpolation
    mode 4: fourth-order interpolation
    '''

    def __init__(self, img_size, s_s=1, s_r=16, order=1, mode='replicate'):

        super(bgridSplatting2DTo3D, self).__init__()
        self.img_size = img_size
        self.s_s = s_s
        self.s_r = int(s_r)
        self.order = order
        self.mode = mode

        h,w = img_size
        gh, gw = int(h*s_s), int(w*s_s)
        self.out_shape = (gh,gw,s_r)

        hg, wg = torch.meshgrid([torch.arange(0,h),torch.arange(0,w)])
        hg = hg.float() / (h-1) * (gh-1)
        wg = wg.float() / (w-1) * (gw-1)
        ones = torch.ones(1,1,h,w,1).float()
        self.register_buffer('hg', hg)
        self.register_buffer('wg', wg)
        self.register_buffer('ones', ones)

    def forward(self, x, gm):

        '''
        x: (n,c,gh,gw)
        gm: (n,1,h,w)
        '''
        n,c,h,w = x.shape
        if gm.shape[2] != h or gm.shape[3] != w:
            gm = nnf.interpolate(gm, (h,w), mode='bilinear', align_corners=True) # n,1,h,w
        out_shape = self.out_shape
        gm = gm * (self.s_r-1)

        hg = self.hg.view(1,h,w).repeat(n,1,1)
        wg = self.wg.view(1,h,w).repeat(n,1,1)
        ones = self.ones.repeat(n,1,1,1,1)

        x = x.unsqueeze(-1) # n,c,h,w,1
        x = torch.cat([x, ones], dim=1) # n,c+1,gh,gw,1
        gm = gm[:,0] # n,gh,gw
        gm = torch.stack([hg, wg, gm], dim=-1).unsqueeze(-2) # n,gh,gw,1,3
        bg = grid_push(x, gm, out_shape, order=self.order, mode=self.mode) # n,c+1,gh,gw,s_r

        on = bg[:,-1:]
        bg = bg[:,:-1]

        return bg, on

if __name__ == "__main__":

    pass