"""
Функции потерь
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss для борьбы с дисбалансом классов
    """
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy с Label Smoothing
    """
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        nll_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_criterion(criterion_name='CrossEntropy', **kwargs):
    """
    Factory функция для получения функции потерь
    
    Args:
        criterion_name: название функции ('CrossEntropy', 'Focal', 'LabelSmoothing')
        **kwargs: дополнительные параметры
    
    Returns:
        функция потерь
    """
    if criterion_name == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif criterion_name == 'Focal':
        return FocalLoss(**kwargs)
    elif criterion_name == 'LabelSmoothing':
        return LabelSmoothingCrossEntropy(**kwargs)
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")