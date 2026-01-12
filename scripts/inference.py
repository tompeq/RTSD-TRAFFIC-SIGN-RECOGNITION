"""
Скрипт для предсказаний
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import torch
from PIL import Image
import json
import argparse
from models import create_model
from dataset import Transforms
from utils.config import Config


def load_model(model_path: str, device: torch.device):
    """Загрузка модели"""
    model = create_model(num_classes=Config.NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_labels():
    """Загрузка названий классов"""
    with open(Config.LABEL_MAP, 'r') as f:
        label_map = json.load(f)
    idx_to_class = {v-1: k for k, v in label_map.items()}
    return idx_to_class


def predict(model, image_path: str, device: torch.device, top_k: int = 5):
    """Предсказание"""
    # Загрузка изображения
    image = Image.open(image_path).convert('RGB')
    
    # Трансформация
    transform = Transforms.get_val_transforms(
        img_size=Config.IMG_SIZE,
        mean=Config.MEAN,
        std=Config.STD
    )
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k)
    
    return top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy(), image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Предсказание класса дорожного знака")
    print("="*60)
    
    # Загрузка
    model = load_model(args.model, Config.DEVICE)
    idx_to_class = load_labels()
    
    # Предсказание
    probs, indices, image = predict(model, args.image, Config.DEVICE, args.top_k)
    
    # Результаты
    print(f"\n📊 Топ-{args.top_k} предсказаний:")
    print("-" * 60)
    for i, (prob, idx) in enumerate(zip(probs, indices), 1):
        class_name = idx_to_class.get(idx, f"Unknown_{idx}")
        print(f"{i}. {class_name:15} | {prob*100:6.2f}%")
    print("-" * 60)
    
    print(f"\n✓ Распознанный знак: {idx_to_class[indices[0]]}")
    print(f"  Уверенность: {probs[0]*100:.2f}%")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()