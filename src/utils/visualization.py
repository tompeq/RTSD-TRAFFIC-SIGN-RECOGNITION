"""
Визуализация результатов обучения
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from typing import List, Optional


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: Optional[Path] = None
):
    """
    Построение графиков истории обучения
    
    Args:
        train_losses: потери на обучении
        val_losses: потери на валидации
        train_accs: точность на обучении
        val_accs: точность на валидации
        save_path: путь для сохранения
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # График loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Эпоха', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('История Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # График accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Эпоха', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('История Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 График сохранен: {save_path}")
    
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    top_k: int = 20
):
    """
    Построение матрицы ошибок
    
    Args:
        y_true: истинные метки
        y_pred: предсказанные метки
        class_names: названия классов
        save_path: путь для сохранения
        top_k: количество топ классов для отображения
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Берем топ-K самых частых классов
    class_counts = np.bincount(y_true)
    top_classes = np.argsort(class_counts)[-top_k:]
    
    cm_top = cm[top_classes][:, top_classes]
    labels_top = [class_names[i] for i in top_classes]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_top,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels_top,
        yticklabels=labels_top
    )
    plt.title(f'Матрица ошибок (Топ-{top_k} классов)', fontsize=14, fontweight='bold')
    plt.ylabel('Истинный класс', fontsize=12)
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Матрица ошибок сохранена: {save_path}")
    
    plt.close()


def visualize_predictions(
    images: np.ndarray,
    true_labels: List[int],
    pred_labels: List[int],
    class_names: List[str],
    n_samples: int = 9
):
    """
    Визуализация предсказаний модели
    
    Args:
        images: массив изображений
        true_labels: истинные метки
        pred_labels: предсказанные метки
        class_names: названия классов
        n_samples: количество образцов
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(min(n_samples, len(images))):
        ax = axes[i]
        ax.imshow(images[i])
        
        true_name = class_names[true_labels[i]]
        pred_name = class_names[pred_labels[i]]
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        ax.set_title(f'True: {true_name}\nPred: {pred_name}', 
                    color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()