"""
ИСПРАВЛЕННЫЙ RTSD Dataset для работы с COCO форматом
Замените src/dataset/rtsd_dataset.py этим кодом
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path
from typing import Optional, Callable
import warnings


class RTSDDataset(Dataset):
    """
    PyTorch Dataset для RTSD в формате COCO
    """
    
    def __init__(
        self,
        anno_path: str,
        data_dir: str,
        label_map_path: str,
        transform: Optional[Callable] = None,
        crop_signs: bool = True
    ):
        """
        Args:
            anno_path: путь к COCO JSON (train_anno.json, val_anno.json)
            data_dir: корневая директория с изображениями
            label_map_path: путь к label_map.json
            transform: torchvision transforms
            crop_signs: вырезать знаки по bbox
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.crop_signs = crop_signs
        
        # Загрузка label_map
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.label_map = json.load(f)
        
        print(f"Загружено {len(self.label_map)} классов")
        
        # Загрузка COCO аннотаций
        print(f"Загрузка COCO аннотаций из {anno_path}...")
        with open(anno_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # Парсинг COCO формата
        self.images = {img['id']: img for img in coco_data.get('images', [])}
        self.annotations = coco_data.get('annotations', [])
        self.categories = {cat['id']: cat for cat in coco_data.get('categories', [])}
        
        print(f"  Изображений: {len(self.images)}")
        print(f"  Аннотаций: {len(self.annotations)}")
        print(f"  Категорий: {len(self.categories)}")
        
        # Создаем маппинг category_id -> class_name
        self.cat_id_to_class = {}
        for cat_id, cat_info in self.categories.items():
            # В COCO categories есть 'sign_class' или 'name'
            class_name = cat_info.get('sign_class') or cat_info.get('name')
            if class_name and class_name in self.label_map:
                self.cat_id_to_class[cat_id] = class_name
        
        print(f"  Найдено классов в категориях: {len(self.cat_id_to_class)}")
        
        # Фильтруем аннотации - оставляем только с известными классами
        self.samples = []
        for ann in self.annotations:
            cat_id = ann.get('category_id')
            img_id = ann.get('image_id')
            
            if cat_id in self.cat_id_to_class and img_id in self.images:
                class_name = self.cat_id_to_class[cat_id]
                image_info = self.images[img_id]
                
                # COCO bbox формат: [x, y, width, height]
                bbox_coco = ann.get('bbox', [])
                if len(bbox_coco) == 4:
                    x, y, w, h = bbox_coco
                    bbox = {
                        'xmin': x,
                        'ymin': y,
                        'xmax': x + w,
                        'ymax': y + h
                    }
                else:
                    bbox = None
                
                self.samples.append({
                    'image_path': image_info['file_name'],
                    'class_name': class_name,
                    'label': self.label_map[class_name],
                    'bbox': bbox
                })
        
        self.num_samples = len(self.samples)
        self.num_classes = len(self.label_map)
        
        print(f"✓ Подготовлено {self.num_samples} образцов для обучения\n")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Полный путь к изображению
        img_path = self.data_dir / sample['image_path']
        
        # Загрузка изображения
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            warnings.warn(f"Ошибка загрузки {img_path}: {e}")
            # Пробуем следующий
            return self.__getitem__((idx + 1) % len(self))
        
        # Вырезание знака
        if self.crop_signs and sample['bbox'] is not None:
            image = self._crop_sign(image, sample['bbox'])
        
        # Трансформации
        if self.transform:
            image = self.transform(image)
        
        # 0-based индексация (label_map начинается с 1)
        label = sample['label'] - 1
        
        return image, label
    
    def _crop_sign(self, image, bbox):
        """Вырезание знака по bbox"""
        try:
            xmin = int(bbox['xmin'])
            ymin = int(bbox['ymin'])
            xmax = int(bbox['xmax'])
            ymax = int(bbox['ymax'])
            
            # Проверка координат
            width, height = image.size
            xmin = max(0, min(xmin, width))
            ymin = max(0, min(ymin, height))
            xmax = max(0, min(xmax, width))
            ymax = max(0, min(ymax, height))
            
            if xmax > xmin and ymax > ymin:
                return image.crop((xmin, ymin, xmax, ymax))
        except Exception as e:
            warnings.warn(f"Ошибка вырезания bbox: {e}")
        
        return image
    
    def get_class_distribution(self):
        """Распределение классов"""
        distribution = {}
        for sample in self.samples:
            cls = sample['class_name']
            distribution[cls] = distribution.get(cls, 0) + 1
        return distribution


def create_dataloaders(train_dataset, val_dataset, batch_size=32, num_workers=0, pin_memory=False):
    """Создание DataLoader'ов"""
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader


# Тестирование
if __name__ == "__main__":
    from pathlib import Path
    
    # Пути
    data_dir = Path("data/rtsd-dataset")
    train_anno = Path("data/train_anno.json")
    label_map = Path("data/label_map.json")
    
    # Проверка что файлы существуют
    print("Проверка файлов:")
    print(f"  data_dir: {data_dir.exists()}")
    print(f"  train_anno: {train_anno.exists()}")
    print(f"  label_map: {label_map.exists()}")
    
    if train_anno.exists() and label_map.exists():
        # Создание датасета
        dataset = RTSDDataset(
            anno_path=str(train_anno),
            data_dir=str(data_dir),
            label_map_path=str(label_map),
            transform=None,
            crop_signs=True
        )
        
        print(f"\n✓ Датасет создан!")
        print(f"  Размер: {len(dataset)}")
        
        if len(dataset) > 0:
            # Пробуем загрузить первый образец
            try:
                image, label = dataset[0]
                print(f"\n✓ Первый образец загружен!")
                print(f"  Image type: {type(image)}")
                print(f"  Label: {label}")
            except Exception as e:
                print(f"\n✗ Ошибка загрузки образца: {e}")