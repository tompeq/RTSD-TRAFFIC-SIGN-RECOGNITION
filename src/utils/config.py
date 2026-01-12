"""
Конфигурация проекта
Все настройки в одном месте
"""

import torch
import os
from pathlib import Path


class Config:
    """Основная конфигурация проекта"""
    
    # ==================== ПУТИ ====================
    ROOT_DIR = Path(__file__).parent.parent.parent
    
    # Данные
    DATA_DIR = ROOT_DIR / 'data'
    RTSD_DIR = DATA_DIR / 'rtsd-dataset'
    
    TRAIN_ANNO = DATA_DIR / 'train_anno.json'
    VAL_ANNO = DATA_DIR / 'val_anno.json'
    LABEL_MAP = DATA_DIR / 'label_map.json'
    LABELS_TXT = DATA_DIR / 'labels.txt'
    
    # Выходные директории
    CHECKPOINT_DIR = ROOT_DIR / 'checkpoints'
    RESULTS_DIR = ROOT_DIR / 'results'
    LOGS_DIR = ROOT_DIR / 'logs'
    
    # ==================== ПАРАМЕТРЫ МОДЕЛИ ====================
    NUM_CLASSES = 156
    MODEL_NAME = 'resnet50'
    PRETRAINED = True
    DROPOUT = 0.5
    
    # ==================== ПАРАМЕТРЫ ОБУЧЕНИЯ ====================
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Scheduler
    SCHEDULER_TYPE = 'ReduceLROnPlateau'
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    
    # Early Stopping
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10
    
    # ==================== ПАРАМЕТРЫ ДАННЫХ ====================
    IMG_SIZE = 224
    CROP_SIGNS = True
    
    # Нормализация ImageNet
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # DataLoader
    NUM_WORKERS = 0  # Для Windows!
    PIN_MEMORY = False
    
    # ==================== АУГМЕНТАЦИИ ====================
    TRAIN_AUGMENTATIONS = {
        'rotation': 20,
        'color_jitter': {
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.3,
            'hue': 0.1
        },
        'horizontal_flip': 0.3,
        'affine': {
            'degrees': 0,
            'translate': (0.1, 0.1),
            'scale': (0.9, 1.1)
        }
    }
    
    # ==================== УСТРОЙСТВО ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ==================== СОХРАНЕНИЕ ====================
    SAVE_FREQUENCY = 10
    BEST_MODEL_NAME = 'best_model.pth'
    LAST_MODEL_NAME = 'last_model.pth'
    
    # ==================== ЛОГИРОВАНИЕ ====================
    LOG_INTERVAL = 10
    
    # ==================== ВОСПРОИЗВОДИМОСТЬ ====================
    SEED = 42
    
    @classmethod
    def create_dirs(cls):
        """Создание необходимых директорий"""
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Вывод конфигурации"""
        print("\n" + "="*60)
        print("КОНФИГУРАЦИЯ ПРОЕКТА")
        print("="*60)
        print(f"\n📁 Пути:")
        print(f"  Данные: {cls.DATA_DIR}")
        print(f"  Checkpoint: {cls.CHECKPOINT_DIR}")
        
        print(f"\n🧠 Модель:")
        print(f"  Архитектура: {cls.MODEL_NAME}")
        print(f"  Классов: {cls.NUM_CLASSES}")
        
        print(f"\n🎯 Обучение:")
        print(f"  Batch size: {cls.BATCH_SIZE}")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Learning rate: {cls.LEARNING_RATE}")
        
        print(f"\n💻 Устройство: {cls.DEVICE}")
        if cls.DEVICE.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print("\n" + "="*60 + "\n")