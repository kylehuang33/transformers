import torch
import numpy as np
import torch.nn as nn
from ._bgrid.dags import grid_push
# Note: video_rope functions are imported but not used in the current implementation
# If needed in the future, implement or copy video_rope.py from internvl


class mlp3(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, drop=0.1):
        super(mlp3, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):

        return self.mlp(x)

class mlp2(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, drop=0.1):
        super(mlp2, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):

        return self.mlp(x)

class mlp1(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(mlp1, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
        )

    def forward(self, x):

        return self.mlp(x)

class MLP_JL(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP_JL, self).__init__()
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        # JL-style random Gaussian initialization:
        # each entry ~ N(0, 1/out_dim)
        with torch.no_grad():
            self.mlp.weight.normal_(mean=0.0, std=1.0 / (self.mlp.out_features ** 0.5))

    def forward(self, x):
        return self.mlp(x)

def construct_3d_coords(t_len, h_len, w_len, device):
    t_coords = torch.arange(t_len, device=device)
    h_coords = torch.arange(h_len, device=device)
    w_coords = torch.arange(w_len, device=device)

    grid_t, grid_h, grid_w = torch.meshgrid(t_coords, h_coords, w_coords, indexing='ij')
    grid_t = grid_t / (t_len - 1)
    grid_h = grid_h / (h_len - 1)
    grid_w = grid_w / (w_len - 1)
    
    coords = torch.stack([grid_t, grid_h, grid_w], dim=-1).reshape(-1, 3)  # (t_len*h_len*w_len, 3)
    
    return coords

@torch.no_grad()
def pca_project_block(X, k, center=True, whiten=True, eps=1e-8):
    if center:
        X = X - X.mean(dim=0, keepdim=True)

    # Compute SVD (PCA)
    U, S, Vh = torch.linalg.svd(X.float(), full_matrices=False)   # X = U Σ V^T
    S = S.to(X.dtype)
    Vh = Vh.to(X.dtype)

    W = Vh[:k, :]                                         # top-k principal directions
    Z = X @ W.t()                                         # projected data (L, k)

    if whiten:
        Z = Z / (S[:k].unsqueeze(0) + eps)

    return Z, W

class bgridVideoTokenCompress(nn.Module):

    def __init__(self, 
        img_size=(16,16), 
        s_r=4, 
        gm_cs=4, 
        s_s=1,
        llm_hidden_size=4096,
        base_t=1000.0,
        base_s=10000.0,
        use_3d_rope=False,
        use_value_mlp=False,
        alpha=0,
        use_key_grad=False,
        use_key_ensemble=1,
        ):

        super(bgridVideoTokenCompress, self).__init__()
        self.img_size = img_size
        self.s_r = int(s_r)
        self.s_s = int(s_s)
        self.gm_cs = gm_cs
        self.llm_hidden_size = llm_hidden_size
        self.use_3d_rope = use_3d_rope
        self.use_value_mlp = use_value_mlp
        self.alpha = alpha
        self.use_key_grad = use_key_grad
        self.use_key_ensemble = use_key_ensemble

        assert self.s_s >= 1, "s_s should be at least 1"
        assert self.s_r >= 2, "s_r should be at least 2"
        assert self.use_key_ensemble >= 1, "use_key_ensemble should be at least 1"

        h,w = img_size 
        if self.s_s > 1:
            self.out_shape = [self.s_s] * 2 + [self.s_r for _ in range(gm_cs)]
        else:
            self.out_shape = [self.s_r for _ in range(gm_cs)]
        ones = torch.ones(1,1,h,w).float()
        self.register_buffer('ones', ones)

        self.mlp_pre_key = nn.Identity()
        self.mlp_post_key = nn.ModuleList([mlp1(llm_hidden_size+3, gm_cs).to(torch.bfloat16) for _ in range(use_key_ensemble)])
        if not self.use_key_grad:
            for param in self.mlp_post_key.parameters():
                param.requires_grad = False
            print("Key MLP gradients are disabled.")
        
        if self.use_value_mlp:
            self.mlp_pre_val = mlp2(llm_hidden_size, llm_hidden_size, llm_hidden_size).to(torch.bfloat16)
            self.mlp_post_val = mlp2(llm_hidden_size, llm_hidden_size, llm_hidden_size).to(torch.bfloat16)
        else:
            self.mlp_pre_val = nn.Identity()
            self.mlp_post_val = nn.Identity()

    def get_post_key(self, key):
        
        key_sum = 0
        for i in range(self.use_key_ensemble):
            key_i = self.mlp_post_key[i](key)  # T,h*w,gm_cs
            key_sum += key_i
        key = key_sum / self.use_key_ensemble
        
        return key

    def packed_key_process_no_rope(self, vit_embeds, thw_coords, accum_list):

        h,w = self.img_size
        c = vit_embeds.shape[-1]

        if self.alpha == 0:
            key = self.mlp_pre_key(vit_embeds)
            # key = torch.cat([key, thw_coords], dim=-1)  # T,h*w,llm_hidden_size+3
            if thw_coords is not None:
                # Make sure coordinates live on the same device (and dtype) as key
                thw_coords = thw_coords.to(key.device, dtype=key.dtype)
                thw_coords = thw_coords.to(key.dtype)

                key = torch.cat([key, thw_coords], dim=-1)  # T, H*W, hidden_dim+3
            key = self.get_post_key(key)  # T,h*w,gm_cs
            key = key.permute(0,2,1).contiguous().view(vit_embeds.shape[0], -1, h, w)  # T,gm_cs,h,w
            return None, key
        key = self.mlp_pre_key(vit_embeds)  # T,h*w,llm_hidden_size
        key = torch.cat([key, thw_coords], dim=-1)  # T,h*w,llm_hidden_size+3

        pre_key1 = key.clone()  # T,h*w,llm_hidden_size+3
        pre_key2 = key.clone()  # T,h*w,llm_hidden_size+3

        key_list = []
        for i in range(len(accum_list) - 1):
            st = accum_list[i]
            ed = accum_list[i+1]
            key_i = pre_key1[st:ed]  # T_i,h*w,llm_hidden_size
            key_i = key_i.view(-1, c+3)  # T_i*h*w, llm_hidden_size
            key_i = pca_project_block(key_i, k=self.gm_cs, center=True, whiten=True)[0]  # T_i*h*w, gm_cs
            key_i = key_i.view(-1, h*w, self.gm_cs)  # T_i,h*w, gm_cs
            key_list.append(key_i)
        key1 = torch.cat(key_list, dim=0)  # T,h*w,gm_cs
        # ------- tmp remove ------- #        
        key2 = self.get_post_key(pre_key2)  # T,h*w,gm_cs
        # ------- tmp remove ------- #
        
        key1 = key1.permute(0,2,1).contiguous().view(vit_embeds.shape[0], -1, h, w)  # T,gm_cs,h,w 
        key2 = key2.permute(0,2,1).contiguous().view(vit_embeds.shape[0], -1, h, w)  # T,gm_cs,h,w

        return key1, key2

    def min_max_normalize_key(self, key, accum_list):

        h,w = self.img_size
        new_keys = []
        for i in range(len(accum_list) - 1):
            st = accum_list[i]
            ed = accum_list[i+1]
            key_i = key[st:ed]  # T_i,gm_cs,h,w
            key_i = key_i.permute(1,0,2,3).contiguous().view(key_i.shape[1], -1)  # gm_cs, T_i*h*w
            key_i = (key_i - key_i.min(dim=-1, keepdim=True)[0]) / (key_i.max(dim=-1, keepdim=True)[0] - key_i.min(dim=-1, keepdim=True)[0] + 1e-8) # T_i*h*w,gm_cs
            key_i = key_i.view(key_i.shape[0], -1, h, w).contiguous().permute(1,0,2,3)  # T_i,gm_cs,h,w
            new_keys.append(key_i)
        key = torch.cat(new_keys, dim=0)  # T,gm_cs,h,w
        return key

    def forward(self, 
            vit_embeds, 
            num_patches_list=[], 
            thw_coords=None, 
            print_key_distribution=False, 
            key_optimize=False, 
            vit_embeds_premlp=None,
        ):

        '''
        vit_embeds: (T,h*w,c)
        thw_coords: (T,h*w,3)
        '''
        T = vit_embeds.shape[0]
        h,w = self.img_size
        L = T*h*w

        accum_list = [0]
        for npatch in num_patches_list:
            accum_list.append(accum_list[-1] + npatch)

        if vit_embeds_premlp is None:
            vit_embeds_premlp = vit_embeds

        key1, key2 = self.packed_key_process_no_rope(vit_embeds_premlp, thw_coords, accum_list)  # T,gm_cs,h,w
        if key1 is None:
            key = self.min_max_normalize_key(key2, accum_list)  # T,gm_cs,h,w
        else:
            key1 = self.min_max_normalize_key(key1, accum_list)  # gram key
            key2 = self.min_max_normalize_key(key2, accum_list)  # mlp key
            key = self.alpha * key1 + (1 - self.alpha) * key2  # T,gm_cs,h,w
        post_key = key.clone().view(T, -1, h*w).contiguous().permute(0,2,1)  # T,L,gm_cs

        if key_optimize:
            return None, post_key
        key = key* (self.s_r - 1)  # scale to [0,s_r-1]

        if print_key_distribution:
            tmp_key = key.permute(0,2,3,1).contiguous().view(-1, key.shape[1])  # (T*h*w),gm_cs
            bins = np.arange(-0.5, self.s_r, 1) 
            for gm_i in range(key.shape[1]):
                key_i = tmp_key[:,gm_i].detach().cpu().float().numpy()
                # histogram bins
                counts, edges = np.histogram(key_i, bins=bins)
                print(f'Key distribution for gm channel {gm_i}: counts={counts}, edges={edges}')
        if self.s_s > 1:
            thw_coords = thw_coords.permute(0,2,1).contiguous()[:,1:].view(T, 2, h, w)  # T,2,h,w
            thw_coords = thw_coords * (self.s_s - 1)
            key = torch.cat([thw_coords, key], dim=1)  # T,gm_cs+2,h,w

        # prepare value
        val = self.mlp_pre_val(vit_embeds) # T,seq_len,c
        val = val.permute(0,2,1).view(T, -1, h, w) # T,c,h,w

        # FIXME: replace
        # out_shape = self.out_shape
        # ones = self.ones.repeat(T,1,1,1)

        # val = torch.cat([val, ones], dim=1) # n,c+1,gh,gw
        # pos_val = [v for v in val.shape] + [1] * (len(out_shape)-2)
        # val = val.view(pos_val) # n,c+1,h,w,1,...,1
        
        # key = key.permute(0,2,3,1) # n,gh,gw,gm_cs
        # pos_key = [v for v in key.shape[:3]] + [1] * (len(out_shape)-2) + [key.shape[-1]]
        # key = key.view(pos_key) # n,gh,gw,1,...,1,gm_cs
        # bg = grid_push(val, key, out_shape, order=1, mode='replicate') # n,c+1,gh,gw,s_r,...,s_r

        out_shape = self.out_shape

        # Make sure the broadcasted ones are on the same device/dtype as val
        ones = self.ones.to(device=val.device, dtype=val.dtype).repeat(T, 1, 1, 1)

        val = torch.cat([val, ones], dim=1)  # n, c+1, gh, gw
        pos_val = [v for v in val.shape] + [1] * (len(out_shape) - 2)
        val = val.view(pos_val)  # n, c+1, h, w, 1, ..., 1
        
        
        key = key.permute(0, 2, 3, 1)       # n,gh,gw,gm_cs
        pos_key = [v for v in key.shape[:3]] + [1] * (len(out_shape) - 2) + [key.shape[-1]]
        key = key.view(pos_key)             # n,gh,gw,1,...,1,gm_cs

        # -------- NEW: force grid_push to run on cuda:0 --------
        target_device = torch.device("cuda:0")   # the GPU bgrid likes to use internally

        orig_device = val.device                # e.g. cuda:3 when model is sharded
        val_gpu0 = val.to(orig_device)
        key_gpu0 = key.to(orig_device)

        bg = grid_push(val_gpu0, key_gpu0, out_shape, order=1, mode='replicate')

        # move result back to original device for the rest of the model
        # bg = bg.to(orig_device)

        ######
        
        


        on = bg[:,-1:]
        bg = bg[:,:-1]
        compressed_list = []
        for i in range(len(num_patches_list)):
            st = accum_list[i]
            ed = accum_list[i+1]
            bg_i = bg[st:ed].sum(dim=0, keepdim=True)  # 1,c,s_r,...,s_r
            on_i = on[st:ed].sum(dim=0, keepdim=True)  # 1,1,s_r,...,s_r
            bg_i = bg_i / (on_i + 1e-8)

            compressed_list.append(bg_i)

        bg = torch.cat(compressed_list, dim=0)  # n,c,s_r,...,s_r
        bg = bg.view(bg.shape[0], self.llm_hidden_size, -1).contiguous().permute(0,2,1) # n,L,c
        bg = self.mlp_post_val(bg) # n,L,c

        return bg, post_key

class GramMatrixLoss(nn.Module):
    """
    CKA-style centered Gram loss (no normalization).
    Inputs x, y have shape (T, L, C).
    We compare centered Gram matrices over the L dimension for each T.
    """
    def __init__(self, weight=0.01):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        self.weight = weight

    @staticmethod
    def _center_along_L(z):
        # z: (T, L, C). Center along L (token/time) axis.
        return z - z.mean(dim=1, keepdim=True)

    def forward(self, x, y):
        """
        x, y: (T, L, C)
        Returns: scalar loss
        """
        # Center features along L (CKA-style centering)
        x_c = self._center_along_L(x)   # (T, L, C)
        y_c = self._center_along_L(y)   # (T, L, C)

        # Centered Gram matrices over L: G = X_c X_c^T -> (T, L, L)
        Gx = torch.matmul(x_c, x_c.transpose(-1, -2))
        Gy = torch.matmul(y_c, y_c.transpose(-1, -2))

        # No normalization (as requested): plain MSE between centered Grams
        loss = self.loss_fn(Gx, Gy)
        return loss * self.weight

if __name__ == "__main__":
    # simple test
    T, c, h, w = 4, 4096, 16, 16
    gm_cs = 8
    img_size = (h,w)
    vit_embeds = torch.randn(T, h*w, c)

    model = bgridVideoTokenCompress(img_size, gm_cs=gm_cs, s_r=2)
    vit_embeds = model(vit_embeds)
    print(vit_embeds.shape)