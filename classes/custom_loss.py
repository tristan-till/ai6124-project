import torch
import torch.nn as nn

class HitRateLoss(nn.Module):
    def __init__(self, base_loss_fn=nn.MSELoss(), alpha=0.5):
        """
        Custom loss criterion integrating hit rate.
        
        Parameters:
        - base_loss_fn: The primary loss function (e.g., MSELoss, L1Loss).
        - alpha: Weight for the hit rate penalty/reward term. A value between 0 and 1.
        """
        super(HitRateLoss, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.alpha = alpha

    def forward(self, predictions, targets):
        base_loss = self.base_loss_fn(predictions, targets)

        prediction_signs = torch.sign(predictions)  
        target_signs = torch.sign(targets)          
        hit_mask = (prediction_signs == target_signs).float()
        hit_rate = hit_mask.mean()                 
        penalty = 1 - hit_rate                     

        combined_loss = base_loss + self.alpha * penalty

        return combined_loss
