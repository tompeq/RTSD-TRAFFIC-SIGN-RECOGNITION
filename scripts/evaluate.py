"""
Скрипт для оценки модели
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import classification_report
from models import create_model
from dataset import RTSDDataset, get_transforms_from_config
from utils.config import Config
from utils.visualization import plot_confusion_matrix


def evaluate_model(model, dataloader, device):
    """Оценка модели"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluation'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def main():
    print("\n" + "="*60)
    print("Оценка модели на валидационном наборе")
    print("="*60 + "\n")
    
    # Датасет
    val_transform = get_transforms_from_config(Config, mode='val')
    val_dataset = RTSDDataset(
        anno_path=str(Config.VAL_ANNO),
        data_dir=str(Config.RTSD_DIR),
        label_map_path=str(Config.LABEL_MAP),
        transform=val_transform,
        crop_signs=Config.CROP_SIGNS
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    # Модель
    model = create_model(num_classes=Config.NUM_CLASSES)
    checkpoint = torch.load(Config.CHECKPOINT_DIR / Config.BEST_MODEL_NAME, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(Config.DEVICE)
    
    # Оценка
    preds, labels = evaluate_model(model, val_loader, Config.DEVICE)
    
    # Метрики
    accuracy = (preds == labels).mean() * 100
    print(f"\n📊 Accuracy: {accuracy:.2f}%\n")
    
    # Загрузка имен классов
    with open(Config.LABELS_TXT, 'r') as f:
        class_names = [line.strip() for line in f]
    
    # Отчет
    print(classification_report(labels, preds, target_names=class_names, zero_division=0))
    
    # Матрица ошибок
    plot_confusion_matrix(
        labels,
        preds,
        class_names,
        save_path=Config.RESULTS_DIR / 'confusion_matrix.png'
    )
    
    print("\n✓ Оценка завершена!")


if __name__ == "__main__":
    main()