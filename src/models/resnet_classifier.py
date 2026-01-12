"""
ResNet классификатор
"""

import torch
import torch.nn as nn
from torchvision import models
from .base_model import BaseModel


class ResNetClassifier(BaseModel):
    """Классификатор на основе ResNet"""
    
    def __init__(
        self,
        num_classes: int = 156,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super(ResNetClassifier, self).__init__(num_classes)
        
        self.backbone_name = backbone
        self.dropout_rate = dropout
        
        # Загрузка backbone
        self.backbone = self._get_backbone(backbone, pretrained)
        
        # Замена последнего слоя
        num_features = self.backbone.fc.in_features
        self.backbone.fc = self._create_classifier(num_features, num_classes, dropout)
    
    def _get_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Загрузка backbone"""
        models_dict = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet152': models.resnet152
        }
        
        if backbone not in models_dict:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        return models_dict[backbone](pretrained=pretrained)
    
    def _create_classifier(self, num_features: int, num_classes: int, dropout: float) -> nn.Module:
        """Создание классификационной головы"""
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.6),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def create_model(
    num_classes: int = 156,
    model_name: str = 'resnet50',
    pretrained: bool = True,
    dropout: float = 0.5
) -> BaseModel:
    """Factory функция для создания модели"""
    return ResNetClassifier(
        num_classes=num_classes,
        backbone=model_name,
        pretrained=pretrained,
        dropout=dropout
    )