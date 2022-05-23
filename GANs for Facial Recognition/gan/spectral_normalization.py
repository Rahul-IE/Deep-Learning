import torch
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        """
        Reference:
        SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS: https://arxiv.org/pdf/1802.05957.pdf
        """
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _update_u_v(self):
        
        #Attribute Extraction
        u = getattr(self.module, self.name + '_u')
        v = getattr(self.module, self.name + '_v')
        w = getattr(self.module, self.name + '_bar')
        
        #parameters from _make_params function
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        #Power Iteration
        for _ in range(self.power_iterations):
            
            v = l2normalize(torch.mv((w.view(height, width).data).T, u.data))
            u = l2normalize(torch.mv(w.view(height, width).data, v.data))
        
        #w Spec Norm
        sing_w = torch.matmul(u, torch.mv(w.view(height, width), v))
        w_SN = w/(sing_w.expand_as(w))
        
        #Update w
        setattr(self.module, self.name, w_SN)
        
    def _make_params(self):
        """
        v: Initialize v with a random vector (sampled from isotropic distrition).
        u: Initialize u with a random vector (sampled from isotropic distrition).
        w: Weight of the current layer.
        """
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        
        self._update_u_v()
        return self.module.forward(*args)