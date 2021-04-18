import torch
from torch import nn
import torch.nn.functional as F

def mask2weight(mask2d, mask_kernel, padding=0):
    # from the feat2d.py,we can know the mask2d is 4-d
    weight = torch.conv2d(mask2d[None,None,:,:].float(),
        mask_kernel, padding=padding)[0, 0]
    weight[weight > 0] = 1 / weight[weight > 0]
    return weight

class TANPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, k, num_stack_layers, mask2d, freeze, causality, tau, gamma, alpha):
        super(TANPredictor, self).__init__()
        self.causal = causality
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        
        # Padding to ensure the dimension of the output map2d
        mask_kernel = torch.ones(1,1,k,k).to(mask2d.device)
        first_padding = (k - 1) * num_stack_layers // 2

        self.weights = [
            mask2weight(mask2d, mask_kernel, padding=first_padding)
        ]
        self.convs = nn.ModuleList(
            [nn.Conv2d(input_size, hidden_size, k, padding=first_padding)]
        )
 
        for _ in range(num_stack_layers - 1):
            self.weights.append(mask2weight(self.weights[-1] > 0, mask_kernel))
            self.convs.append(nn.Conv2d(hidden_size, hidden_size, k))
            
        
        if self.causal:
            # Causal Inference predictor
            self.pred = nn.Conv2d(hidden_size, 1, 1, bias=False)
            self.register_buffer('d', torch.zeros((1, hidden_size, mask2d.shape[0], mask2d.shape[1]), device='cuda', requires_grad=False))
            self.cos_sim = nn.CosineSimilarity(dim=1)
        else:
            # Original predictor
            self.pred = nn.Conv2d(hidden_size, 1, 1)
        
        if freeze:
            for p in self.convs.parameters():
                p.requires_grad = False
        
    def forward(self, x):
        B, hdim, nclip, nclip = x.size()
        for conv, weight in zip(self.convs, self.weights):
            x = conv(x).relu() * weight
        
        # Casual inference
        if self.causal:
            if self.training:
                # De-confound Training
                self.d = 0.9*self.d + 0.1*x.detach().mean(0, keepdim=True)
                x = F.normalize(x, dim=1)
                x = self.tau * self.pred(x).squeeze_()/(torch.norm(self.pred.weight) + self.gamma)
            else:
                # counterfactual TDE inference
                bias = self.cos_sim(x, self.d).unsqueeze(1) * F.normalize(self.d, dim=1)
                x = F.normalize(x, dim=1)
                x = self.tau * self.pred(x-self.alpha*bias).squeeze_()/(torch.norm(self.pred.weight) + self.gamma)
        else:
            x = self.pred(x).squeeze_()
        return x


def build_predictor(cfg, arch, mask2d):
    causality = cfg.MODEL.CAUSAL.USE_CAUSALITY
    tau = cfg.MODEL.CAUSAL.TAU
    gamma = cfg.MODEL.CAUSAL.GAMMA
    alpha = cfg.MODEL.CAUSAL.ALPHA
    if arch == 'TAN':
        input_size = cfg.MODEL.TAN.FEATPOOL.HIDDEN_SIZE
        hidden_size = cfg.MODEL.TAN.PREDICTOR.HIDDEN_SIZE
        kernel_size = cfg.MODEL.TAN.PREDICTOR.KERNEL_SIZE
        num_stack_layers = cfg.MODEL.TAN.PREDICTOR.NUM_STACK_LAYERS
        freeze = cfg.MODEL.TAN.PREDICTOR.FREEZE
        return TANPredictor(
            input_size, hidden_size, kernel_size, num_stack_layers, mask2d, freeze,
            causality, tau, gamma, alpha
        ) 
    else:
        raise NotImplementedError
