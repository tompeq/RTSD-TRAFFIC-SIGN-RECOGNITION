"""
Базовый класс для всех моделей
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(ABC, nn.Module):
    """Абстрактный базовый класс для моделей классификации"""
    
    def __init__(self, num_classes: int):
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        pass
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Предсказание класса"""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Предсказание вероятностей"""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def get_num_parameters(self) -> int:
        """Количество параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """Заморозка backbone"""
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Разморозка backbone"""
        if hasattr(self, 'backbone'):
            for param in self.backbone.parameters():
                param.requires_grad = True
    
    def summary(self):
        """Вывод информации о модели"""
        print(f"\n{'='*60}")
        print(f"Model: {self.__class__.__name__}")
        print(f"{'='*60}")
        print(f"Classes: {self.num_classes}")
        print(f"Parameters: {self.get_num_parameters():,}")
        print(f"{'='*60}\n")