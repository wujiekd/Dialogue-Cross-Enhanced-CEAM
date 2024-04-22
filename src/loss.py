import torch
import torch.nn as nn

class CenterMSELoss(nn.Module):
    def __init__(self, reduction='none', beta=0.5):
        super(CenterMSELoss, self).__init__()
        self.beta = beta
        self.MSE_loss = nn.MSELoss(reduction=reduction)

    def forward(self, outputs, labels):
        mse_loss = self.MSE_loss(outputs, labels)

        length = outputs.size(1)

        weights = torch.cat([
            torch.ones(length // 3) * self.beta,
            torch.ones(length // 3),
            torch.ones(length - 2 * (length // 3)) * self.beta
        ]).to(outputs.device)
        weights = weights.unsqueeze(0).expand(outputs.shape[0], -1)

        weighted_loss = torch.mean(mse_loss * weights) # explicitly

        weighted_loss = （self.beta+1+self.beta) / 3 * weighted_loss  # explicitly normalize or implicitly

        return weighted_loss

