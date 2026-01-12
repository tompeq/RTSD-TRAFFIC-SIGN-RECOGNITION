"""
Трансформации и аугментации
"""

from torchvision import transforms
import torch


class Transforms:
    """Класс для создания трансформаций"""
    
    @staticmethod
    def get_train_transforms(
        img_size: int = 224,
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225],
        rotation: int = 20,
        color_jitter: dict = None,
        horizontal_flip: float = 0.3,
        affine: dict = None
    ) -> transforms.Compose:
        """Трансформации для обучения"""
        transform_list = [transforms.Resize((img_size, img_size))]
        
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(rotation))
        
        if color_jitter:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=color_jitter.get('brightness', 0.3),
                    contrast=color_jitter.get('contrast', 0.3),
                    saturation=color_jitter.get('saturation', 0.3),
                    hue=color_jitter.get('hue', 0.1)
                )
            )
        
        if horizontal_flip > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=horizontal_flip))
        
        if affine:
            transform_list.append(
                transforms.RandomAffine(
                    degrees=affine.get('degrees', 0),
                    translate=affine.get('translate', (0.1, 0.1)),
                    scale=affine.get('scale', (0.9, 1.1))
                )
            )
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        return transforms.Compose(transform_list)
    
    @staticmethod
    def get_val_transforms(
        img_size: int = 224,
        mean: list = [0.485, 0.456, 0.406],
        std: list = [0.229, 0.224, 0.225]
    ) -> transforms.Compose:
        """Трансформации для валидации"""
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


def get_transforms_from_config(config, mode='train'):
    """Создание трансформаций из конфигурации"""
    if mode == 'train':
        return Transforms.get_train_transforms(
            img_size=config.IMG_SIZE,
            mean=config.MEAN,
            std=config.STD,
            rotation=config.TRAIN_AUGMENTATIONS.get('rotation', 20),
            color_jitter=config.TRAIN_AUGMENTATIONS.get('color_jitter'),
            horizontal_flip=config.TRAIN_AUGMENTATIONS.get('horizontal_flip', 0.3),
            affine=config.TRAIN_AUGMENTATIONS.get('affine')
        )
    else:
        return Transforms.get_val_transforms(
            img_size=config.IMG_SIZE,
            mean=config.MEAN,
            std=config.STD
        )