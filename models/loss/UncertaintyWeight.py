import torch
import torch.nn as nn

class UncertaintyWeightBalancing(nn.Module):
    def __init__(self):
        super().__init__()
        self.gaussian_nll = nn.GaussianNLLLoss(reduction='mean', eps=1e-8)
        self.log_sigma_mlm = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_reg = nn.Parameter(torch.tensor(0.0))

    def forward(self, loss_mlm, eta_output, targets):
        mean = eta_output[:, 0] # (B,)
        log_var = eta_output[:, 1] # (B,)
        var = torch.exp(log_var).clamp(min=1e-6) # (B,)  
        gaussian_nll = self.gaussian_nll(mean, targets.squeeze(), var)

        precision_mlm = torch.exp(-self.log_sigma_mlm)
        precision_reg = torch.exp(-self.log_sigma_reg)

        weighted_mlm = precision_mlm * loss_mlm + self.log_sigma_mlm
        weighted_reg = precision_reg * gaussian_nll + self.log_sigma_reg
        total_loss = weighted_mlm + weighted_reg
        loss_dict = {
            'loss_mlm': loss_mlm,
            'gaussian_nll': gaussian_nll,
            'weighted_mlm': weighted_mlm,
            'weighted_reg': weighted_reg,
            'sigma_mlm': torch.exp(self.log_sigma_mlm),
            'sigma_reg': torch.exp(self.log_sigma_reg)
        }
        return total_loss, loss_dict
