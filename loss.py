import torch
import torch.nn as nn

class MimickingLoss(nn.Module):
    def __init__(self):
        super(MimickingLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, fea_t, fea_s):
        """
        Calculate L_fea according to Eq. (3)[cite: 171].
        L_fea = ||Fea_T - Fea_S||^2_2
        """
        return self.mse(fea_t, fea_s)

def calculate_total_loss(pred, target, fea_t, fea_s, loss_fn_original, lambda_weight=1.0):
    """
    Calculate L_total according to Eq. (4) [cite: 175].
    L_total = L_fea + L_o
    """
    # Original task loss (e.g., reconstruction loss)
    l_o = loss_fn_original(pred, target)
    
    # Feature Mimicking loss
    l_fea = nn.MSELoss()(fea_t, fea_s)
    
    # Simple summation as per Eq. (4)
    l_total = l_fea + l_o
    
    return l_total, l_fea, l_o
