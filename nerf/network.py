import jittor as jt
import jittor.nn as nn
from jittor import Function as F

from activation import trunc_exp
from .renderer import NeRFRenderer

import numpy as np
from encoding import get_encoder

from .utils import safe_normalize

# TODO: not sure about the details...
class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.norm = nn.LayerNorm(self.dim_out)
        self.activation = nn.SiLU()

        if self.dim_in != self.dim_out:
            self.skip = nn.Linear(self.dim_in, self.dim_out, bias=False)
        else:
            self.skip = None

    def execute(self, x):
        # x: [B, C]
        identity = x

        out = self.dense(x)
        out = self.norm(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.activation(out)

        return out

class BasicBlock(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.dense = nn.Linear(self.dim_in, self.dim_out, bias=bias)
        self.activation = nn.ReLU()

    def execute(self, x):
        # x: [B, C]

        out = self.dense(x)
        out = self.activation(out)

        return out    

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, block=BasicBlock):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            if l == 0:
                net.append(BasicBlock(self.dim_in, self.dim_hidden, bias=bias))
            elif l != num_layers - 1:
                net.append(block(self.dim_hidden, self.dim_hidden, bias=bias))
            else:
                net.append(nn.Linear(self.dim_hidden, self.dim_out, bias=bias))

        self.net = nn.ModuleList(net)
        
    
    def execute(self, x):

        for l in range(self.num_layers):
            x = self.net[l](x)
            
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self, 
                 opt,
                 num_layers=4, # 5 in paper
                 hidden_dim=96, # 128 in paper
                 num_layers_bg=2, # 3 in paper
                 hidden_dim_bg=64, # 64 in paper
                 encoding='frequency_jt', # pure pyjt
                 ):
        
        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.encoder, self.in_dim = get_encoder(encoding, input_dim=3, multires=6)
        self.sigma_net = MLP(self.in_dim, 4, hidden_dim, num_layers, bias=True, block=ResBlock)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg   
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding, input_dim=3, multires=4)
            self.bg_net = MLP(self.in_dim_bg, 3, hidden_dim_bg, num_layers_bg, bias=True)
            
        else:
            self.bg_net = None

    def gaussian(self, x):
        # x: [B, N, 3]
        
        d = (x ** 2).sum(-1)
        g = self.opt.blob_density * jt.exp(- d / (self.opt.blob_radius ** 2))

        return g

    def common_forward(self, x):
        # x: [N, 3], in [-bound, bound]

        # sigma
        h = self.encoder(x, bound=self.bound)

        h = self.sigma_net(h)

        sigma = trunc_exp(h[..., 0] + self.gaussian(x))
        albedo = jt.sigmoid(h[..., 1:])

        return sigma, albedo
    
    # ref: https://github.com/zhaofuq/Instant-NSR/blob/main/nerf/network_sdf.py#L192
    def finite_difference_normal(self, x, epsilon=1e-2):
        # x: [N, 3]
        dx_pos, _ = self.common_forward((x + jt.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dx_neg, _ = self.common_forward((x + jt.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_pos, _ = self.common_forward((x + jt.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dy_neg, _ = self.common_forward((x + jt.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))
        dz_pos, _ = self.common_forward((x + jt.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        dz_neg, _ = self.common_forward((x + jt.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))
        
        normal = jt.stack([
            0.5 * (dx_pos - dx_neg) / epsilon, 
            0.5 * (dy_pos - dy_neg) / epsilon, 
            0.5 * (dz_pos - dz_neg) / epsilon
        ], dim=-1)

        return -normal
    
    def normal(self, x):
    
        with jt.enable_grad():
            x.requires_grad = True
            sigma, albedo = self.common_forward(x)
            # query gradient
            normal = - jt.grad(jt.sum(sigma), x, retain_graph=True)[0] # [N, 3]
        
        # normal = self.finite_difference_normal(x)
        normal = safe_normalize(normal)
        # normal = jt.array(normal)

        return normal
        
    def execute(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]
        # ratio: scalar, ambient ratio, 1 == no shading (albedo only), 0 == only shading (textureless)

        if shading == 'albedo':
            # no need to query normal
            #sigma, albedo = self.common_forward(x)
            sigma, color = self.common_forward(x)
            normal=None
            # normal = self.normal(x)
            # color = albedo
        
        else:
            # query normal

            # sigma, albedo = self.common_forward(x)
            # normal = self.normal(x)
        
            with jt.enable_grad():
                x.requires_grad_(True)
                sigma, albedo = self.common_forward(x)
                # query gradient
                normal = - jt.autograd.grad(jt.sum(sigma), x, create_graph=True)[0] # [N, 3]
            normal = safe_normalize(normal)
            # normal = jt.nan_to_num(normal)
            # normal = normal.detach()

            # lambertian shading
            lambertian = ratio + (1 - ratio) * (normal @ l).clamp(min=0) # [N,]

            if shading == 'textureless':
                color = lambertian.unsqueeze(-1).repeat(1, 3)
            elif shading == 'normal':
                color = (normal + 1) / 2
            else: # 'lambertian'
                color = albedo * lambertian.unsqueeze(-1)
            
        return sigma, color, normal

      
    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        
        sigma, albedo = self.common_forward(x)
        
        return {
            'sigma': sigma,
            'albedo': albedo,
        }


    def background(self, d):

        h = self.encoder_bg(d) # [N, C]
        
        h = self.bg_net(h)

        # sigmoid activation for rgb
        rgbs = jt.sigmoid(h)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            # {'params': self.encoder.parameters(), 'lr': lr * 10},
            {'params': self.sigma_net.parameters(), 'lr': lr},
        ]        

        if self.bg_radius > 0:
            # params.append({'params': self.encoder_bg.parameters(), 'lr': lr * 10})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})

        return params