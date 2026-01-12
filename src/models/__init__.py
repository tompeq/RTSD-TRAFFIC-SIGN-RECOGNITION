"""
Модуль моделей
"""

from .base_model import BaseModel
from .resnet_classifier import ResNetClassifier, create_model

__all__ = [
    'BaseModel',
    'ResNetClassifier',
    'create_model'
]