"""
Метрики для оценки модели
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from typing import Tuple, Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Вычисление метрик классификации
    
    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
    
    Returns:
        словарь с метриками
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


class MetricsTracker:
    """Класс для отслеживания метрик во время обучения"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Сброс всех метрик"""
        self.running_loss = 0.0
        self.running_correct = 0
        self.running_total = 0
        self.all_predictions = []
        self.all_labels = []
    
    def update(self, loss: float, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Обновление метрик
        
        Args:
            loss: значение функции потерь
            predictions: предсказания модели
            labels: истинные метки
        """
        self.running_loss += loss
        _, predicted = predictions.max(1)
        self.running_correct += predicted.eq(labels).sum().item()
        self.running_total += labels.size(0)
        
        self.all_predictions.extend(predicted.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
    
    def get_metrics(self, num_batches: int) -> Dict[str, float]:
        """
        Получение средних метрик
        
        Args:
            num_batches: количество батчей
        
        Returns:
            словарь с метриками
        """
        avg_loss = self.running_loss / num_batches
        accuracy = 100.0 * self.running_correct / self.running_total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def get_detailed_metrics(self) -> Dict[str, float]:
        """Получение детальных метрик"""
        metrics = calculate_metrics(
            np.array(self.all_labels),
            np.array(self.all_predictions)
        )
        return metrics


def top_k_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    """
    Вычисление Top-K accuracy
    
    Args:
        output: выход модели [batch_size, num_classes]
        target: истинные метки [batch_size]
        k: количество топ предсказаний
    
    Returns:
        Top-K accuracy
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()