"""
Скрипт для проверки датасета RTSD
ПОЛОЖИТЕ ЭТОТ ФАЙЛ В КОРЕНЬ ПРОЕКТА: check_dataset.py
"""

import json
import os
from pathlib import Path

# Определяем пути напрямую (без импорта config)
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / 'data'
RTSD_DIR = DATA_DIR / 'rtsd-dataset'
TRAIN_ANNO = DATA_DIR / 'train_anno.json'
VAL_ANNO = DATA_DIR / 'val_anno.json'
LABEL_MAP = DATA_DIR / 'label_map.json'
LABELS_TXT = DATA_DIR / 'labels.txt'


def check_files():
    """Проверка наличия всех необходимых файлов"""
    
    print("\n" + "="*70)
    print("🔍 ПРОВЕРКА СТРУКТУРЫ ДАТАСЕТА RTSD")
    print("="*70 + "\n")
    
    # Проверка основных путей
    print("📁 Основные пути:")
    print(f"  ROOT_DIR: {ROOT_DIR}")
    print(f"  Существует: {ROOT_DIR.exists()} ✓" if ROOT_DIR.exists() else "  Существует: ✗")
    
    print(f"\n  DATA_DIR: {DATA_DIR}")
    status = "✓" if DATA_DIR.exists() else "✗"
    print(f"  Существует: {status}")
    
    if not DATA_DIR.exists():
        print("\n  ❌ ОШИБКА: Папка data/ не найдена!")
        print(f"     Создайте: mkdir {DATA_DIR}")
        return False
    
    print(f"\n  RTSD_DIR: {RTSD_DIR}")
    status = "✓" if RTSD_DIR.exists() else "✗"
    print(f"  Существует: {status}")
    
    # Проверка файлов аннотаций
    print("\n" + "-"*70)
    print("📄 Файлы аннотаций и конфигурации:")
    print("-"*70)
    
    files_status = {}
    files_to_check = {
        'train_anno.json': TRAIN_ANNO,
        'val_anno.json': VAL_ANNO,
        'label_map.json': LABEL_MAP,
        'labels.txt': LABELS_TXT
    }
    
    all_exist = True
    for name, path in files_to_check.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        files_status[name] = exists
        all_exist = all_exist and exists
        
        print(f"\n  {status} {name}")
        print(f"     Путь: {path}")
        print(f"     Существует: {exists}")
        
        if exists:
            # Показываем размер файла
            size = path.stat().st_size
            if size < 1024:
                print(f"     Размер: {size} байт")
            elif size < 1024*1024:
                print(f"     Размер: {size/1024:.1f} KB")
            else:
                print(f"     Размер: {size/(1024*1024):.1f} MB")
            
            # Для JSON показываем количество записей
            if name.endswith('.json'):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"     Записей: {len(data)}")
                    
                    # Показываем пример для train_anno
                    if name == 'train_anno.json' and len(data) > 0:
                        first_key = list(data.keys())[0]
                        print(f"\n     Пример записи:")
                        print(f"       Ключ: {first_key}")
                        print(f"       Данные: {str(data[first_key])[:100]}...")
                        
                except Exception as e:
                    print(f"     ✗ Ошибка чтения: {e}")
        else:
            print(f"     ❌ ФАЙЛ НЕ НАЙДЕН!")
    
    # Проверка структуры папок с изображениями
    print("\n" + "-"*70)
    print("📂 Структура папок с изображениями:")
    print("-"*70)
    
    if RTSD_DIR.exists():
        # Список подпапок
        subdirs = [d for d in RTSD_DIR.iterdir() if d.is_dir()]
        print(f"\n  Найдено подпапок: {len(subdirs)}")
        
        if subdirs:
            print("  Подпапки:")
            for subdir in sorted(subdirs)[:10]:
                # Подсчет файлов в подпапке
                files = list(subdir.glob('*'))
                print(f"    - {subdir.name}/ ({len(files)} файлов)")
            if len(subdirs) > 10:
                print(f"    ... и еще {len(subdirs) - 10} папок")
        else:
            print("  ⚠️ Подпапок не найдено!")
        
        # Подсчет изображений
        print("\n  Подсчет изображений:")
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_counts = {}
        total_images = 0
        
        for ext in image_extensions:
            images = list(RTSD_DIR.rglob(f'*{ext}'))
            if images:
                image_counts[ext] = len(images)
                total_images += len(images)
        
        if image_counts:
            for ext, count in image_counts.items():
                print(f"    {ext}: {count} файлов")
            print(f"\n  📊 Всего изображений: {total_images}")
        else:
            print("  ❌ Изображения не найдены!")
    else:
        print("\n  ❌ Папка rtsd-dataset/ не найдена!")
        print(f"     Ожидается: {RTSD_DIR}")
    
    print("\n" + "="*70)
    
    return all_exist


def test_dataset_loading():
    """Попытка загрузки первого образца"""
    
    print("\n" + "="*70)
    print("🧪 ТЕСТ ЗАГРУЗКИ ДАННЫХ")
    print("="*70 + "\n")
    
    # Проверяем train_anno.json
    if not TRAIN_ANNO.exists():
        print("❌ train_anno.json не найден!")
        print(f"   Ожидается: {TRAIN_ANNO}")
        return False
    
    print("1️⃣ Загрузка train_anno.json...")
    try:
        with open(TRAIN_ANNO, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        print(f"   ✓ Загружено записей: {len(train_data)}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        return False
    
    if len(train_data) == 0:
        print("   ❌ Файл пустой!")
        return False
    
    # Проверяем label_map
    print("\n2️⃣ Загрузка label_map.json...")
    try:
        with open(LABEL_MAP, 'r', encoding='utf-8') as f:
            label_map = json.load(f)
        print(f"   ✓ Загружено классов: {len(label_map)}")
        print(f"   Примеры классов: {list(label_map.keys())[:5]}")
    except Exception as e:
        print(f"   ✗ Ошибка: {e}")
        return False
    
    # Проверяем существование первого изображения
    print("\n3️⃣ Проверка первого изображения...")
    first_key = list(train_data.keys())[0]
    first_data = train_data[first_key]
    
    print(f"   Путь в аннотации: {first_key}")
    print(f"   Данные: {first_data}")
    
    # Полный путь к изображению
    img_path = RTSD_DIR / first_key
    print(f"\n   Полный путь: {img_path}")
    print(f"   Существует: {img_path.exists()}")
    
    if img_path.exists():
        try:
            from PIL import Image
            img = Image.open(img_path)
            print(f"   Размер: {img.size}")
            print(f"   Формат: {img.format}")
            print("\n   ✅ Изображение успешно загружено!")
        except ImportError:
            print("   ⚠️ PIL/Pillow не установлен, но файл существует")
        except Exception as e:
            print(f"   ✗ Ошибка открытия: {e}")
    else:
        print(f"\n   ❌ Изображение не найдено!")
        print(f"\n   💡 Возможные причины:")
        print(f"      1. Путь в train_anno.json: '{first_key}'")
        print(f"      2. Ожидаемый файл: {img_path}")
        print(f"      3. Проверьте что изображения в: {RTSD_DIR}")
        return False
    
    # Проверяем класс знака
    if 'objects' in first_data and first_data['objects']:
        obj = first_data['objects'][0]
        if 'class' in obj:
            sign_class = obj['class']
            print(f"\n4️⃣ Класс знака: {sign_class}")
            if sign_class in label_map:
                print(f"   ✓ Класс найден в label_map (индекс: {label_map[sign_class]})")
            else:
                print(f"   ✗ Класс НЕ найден в label_map!")
                return False
    
    print("\n" + "="*70)
    print("✅ ВСЁ РАБОТАЕТ! Датасет готов к использованию")
    print("="*70 + "\n")
    
    return True


def show_solutions():
    """Показать решения если что-то не так"""
    
    print("\n" + "="*70)
    print("💡 ЧТО ДЕЛАТЬ ЕСЛИ ФАЙЛЫ НЕ НАЙДЕНЫ")
    print("="*70 + "\n")
    
    print("1. Скачайте датасет с Kaggle:")
    print("   https://www.kaggle.com/datasets/watchman/rtsd-dataset")
    print()
    
    print("2. Распакуйте в следующую структуру:")
    print()
    print("   rtsd-traffic-sign-recognition/")
    print("   ├── data/")
    print("   │   ├── train_anno.json          ← Из архива Kaggle")
    print("   │   ├── val_anno.json            ← Из архива Kaggle")
    print("   │   ├── label_map.json           ← У вас есть")
    print("   │   ├── labels.txt               ← У вас есть")
    print("   │   └── rtsd-dataset/            ← Из архива Kaggle")
    print("   │       ├── train/")
    print("   │       ├── val/")
    print("   │       └── test/")
    print("   ├── src/")
    print("   ├── scripts/")
    print("   └── ...")
    print()
    
    print("3. Или создайте тестовый датасет для проверки кода:")
    print("   python create_dummy_dataset.py")
    print()
    
    print("="*70 + "\n")


def main():
    """Основная функция"""
    
    print("\n" + "="*70)
    print("🚦 ДИАГНОСТИКА ДАТАСЕТА RTSD")
    print("="*70)
    
    # Проверка файлов
    files_ok = check_files()
    
    # Если файлы на месте, пробуем загрузить
    if files_ok:
        test_dataset_loading()
    else:
        print("\n❌ Не все файлы найдены!")
        show_solutions()


if __name__ == "__main__":
    main()