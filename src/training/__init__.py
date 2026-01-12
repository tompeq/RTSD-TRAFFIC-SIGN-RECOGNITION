"""
Модуль обучения
"""

from .trainer import Trainer
from .losses import get_criterion, FocalLoss, LabelSmoothingCrossEntropy

__all__ = [
    'Trainer',
    'get_criterion',
    'FocalLoss',
    'LabelSmoothingCrossEntropy'
]