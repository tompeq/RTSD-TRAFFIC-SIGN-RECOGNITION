"""
Trainer класс
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple, Any
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.metrics import MetricsTracker


class Trainer:
    """Класс для управления обучением"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[Any] = None,
        checkpoint_dir: Path = Path('checkpoints'),
        early_stopping_patience: int = 10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.early_stopping_patience = early_stopping_patience
        
        # История
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0
        self.current_epoch = 0
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Одна эпоха обучения"""
        self.model.train()
        metrics = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]', ncols=100)
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            metrics.update(loss.item(), outputs, labels)
            
            current_metrics = metrics.get_metrics(len(metrics.all_predictions) // len(labels) + 1)
            pbar.set_postfix({
                'loss': f'{current_metrics["loss"]:.4f}',
                'acc': f'{current_metrics["accuracy"]:.2f}%'
            })
        
        final_metrics = metrics.get_metrics(len(self.train_loader))
        return final_metrics['loss'], final_metrics['accuracy']
    
    def validate(self) -> Tuple[float, float, list, list]:
        """Валидация"""
        self.model.eval()
        metrics = MetricsTracker()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} [Val]', ncols=100)
            
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                metrics.update(loss.item(), outputs, labels)
                
                current_metrics = metrics.get_metrics(len(metrics.all_predictions) // len(labels) + 1)
                pbar.set_postfix({
                    'loss': f'{current_metrics["loss"]:.4f}',
                    'acc': f'{current_metrics["accuracy"]:.2f}%'
                })
        
        final_metrics = metrics.get_metrics(len(self.val_loader))
        return (final_metrics['loss'], final_metrics['accuracy'], 
                metrics.all_predictions, metrics.all_labels)
    
    def train(self, num_epochs: int):
        """Полный цикл обучения"""
        print(f"\n{'='*60}")
        print(f"Начало обучения на {self.device}")
        print(f"Эпох: {num_epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Обучение
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Валидация
            val_loss, val_acc, preds, labels = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Вывод результатов
            print(f"\nЭпоха {epoch}/{num_epochs}:")
            print(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
            
            # Сохранение лучшей модели
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model('best_model.pth')
                print(f"  ✓ Лучшая модель! Acc: {val_acc:.2f}%")
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            
            # Early stopping
            if self.epochs_no_improve >= self.early_stopping_patience:
                print(f"\n⚠ Early stopping на эпохе {epoch}")
                break
            
            # Scheduler
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Checkpoint каждые 10 эпох
            if epoch % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth', epoch)
        
        print(f"\n{'='*60}")
        print(f"Обучение завершено! Лучшая Acc: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
    
    def save_model(self, filename: str):
        """Сохранение модели"""
        path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_acc': self.best_val_acc
        }, path)
    
    def save_checkpoint(self, filename: str, epoch: int):
        """Сохранение checkpoint"""
        path = self.checkpoint_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_acc': self.best_val_acc
        }, path)
        print(f"  💾 Checkpoint: {filename}")