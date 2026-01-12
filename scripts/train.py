"""
Скрипт для запуска обучения
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
import torch.optim as optim
from utils.config import Config
from models import create_model
from dataset import RTSDDataset, create_dataloaders, get_transforms_from_config
from training import Trainer, get_criterion
from utils.visualization import plot_training_history
from utils.logger import setup_logger
import random
import numpy as np


def set_seed(seed: int):
    """Установка seed для воспроизводимости"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    # Конфигурация
    Config.print_config()
    Config.create_dirs()
    set_seed(Config.SEED)
    
    # Логгер
    logger = setup_logger(Config.LOGS_DIR, 'training')
    logger.info("Начало обучения")
    
    # Трансформации
    train_transform = get_transforms_from_config(Config, mode='train')
    val_transform = get_transforms_from_config(Config, mode='val')
    
    # Датасеты
    print("\n📁 Загрузка датасетов...")
    train_dataset = RTSDDataset(
        anno_path=str(Config.TRAIN_ANNO),
        data_dir=str(Config.RTSD_DIR),
        label_map_path=str(Config.LABEL_MAP),
        transform=train_transform,
        crop_signs=Config.CROP_SIGNS
    )
    
    val_dataset = RTSDDataset(
        anno_path=str(Config.VAL_ANNO),
        data_dir=str(Config.RTSD_DIR),
        label_map_path=str(Config.LABEL_MAP),
        transform=val_transform,
        crop_signs=Config.CROP_SIGNS
    )
    
    print(f"✓ Train: {len(train_dataset)} образцов")
    print(f"✓ Val: {len(val_dataset)} образцов")
    
    # DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    # Модель
    print("\n🧠 Создание модели...")
    model = create_model(
        num_classes=Config.NUM_CLASSES,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        dropout=Config.DROPOUT
    )
    model.summary()
    
    # Loss, optimizer, scheduler
    criterion = get_criterion('CrossEntropy')
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=Config.DEVICE,
        scheduler=scheduler,
        checkpoint_dir=Config.CHECKPOINT_DIR,
        early_stopping_patience=Config.EARLY_STOPPING_PATIENCE
    )
    
    # Обучение
    trainer.train(num_epochs=Config.NUM_EPOCHS)
    
    # Графики
    plot_training_history(
        trainer.history['train_loss'],
        trainer.history['val_loss'],
        trainer.history['train_acc'],
        trainer.history['val_acc'],
        save_path=Config.RESULTS_DIR / 'training_history.png'
    )
    
    logger.info(f"Обучение завершено. Лучшая Acc: {trainer.best_val_acc:.2f}%")
    print("\n✓ Готово!")


if __name__ == "__main__":
    main()