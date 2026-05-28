import os.path
import random
import shutil
from pathlib import Path

# Настройки
# Путь к исходному датасету HaGRID
SOURCE_ROOT = Path(r"..\..\HaGRIDv2_dataset_512")

# Куда сохранять готовый dataset
DEST_ROOT = Path(r".\MyDataset")

# Соответствие названий жестов "папка в HaGRID" -> "моя папка"
CLASSES = {
    "fist": "fist",
    "_like": "like",
    "_ok": "ok",
    "_palm": "palm",
    "thumb_index": "thumb_index",
    # "_no_gesture": "no_gesture"
}

# Сколько изображений брать с каждого класса
IMAGES_PER_CLASS = 2000

# Пропорции split'а
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Поддерживаемые форматы
EXTENSIONS = [".jpg"]

# random seed для reproducibility
random.seed(42)


def split_dataset():
    for source_class, target_class in CLASSES.items():
        source_dir = SOURCE_ROOT / source_class
        if not source_dir.exists():
            print(f"[ERROR] Папка не найдена: {source_dir}")
            continue

        # Собираем все изображения
        image_files = []
        for ext in EXTENSIONS:
            image_files.extend(source_dir.glob(f"*{ext}"))
        image_files = list(image_files)
        print(f"\n[{target_class}] Найдено изображений: {len(image_files)}")
        if len(image_files) < IMAGES_PER_CLASS:
            print(f"[WARNING] В классе меньше {IMAGES_PER_CLASS} изображений.")

        # Перемешиваем
        random.shuffle(image_files)

        # Берем только нужное количество
        selected_images = image_files[:IMAGES_PER_CLASS]

        # Разбиение
        train_end = int(IMAGES_PER_CLASS * TRAIN_RATIO)
        val_end = train_end + int(IMAGES_PER_CLASS * VAL_RATIO)

        train_images = selected_images[:train_end]
        val_images = selected_images[train_end:val_end]
        test_images = selected_images[val_end:]

        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }

        # Сохранение
        for split_name, split_images in splits.items():
            save_dir = DEST_ROOT / split_name / target_class
            save_dir.mkdir(parents=True, exist_ok=True)
            for image_path in split_images:
                destination = save_dir / image_path.name
                shutil.copy2(image_path, destination)

        print(
            f"[OK] {target_class}: "
            f"train={len(train_images)}, "
            f"val={len(val_images)}, "
            f"test={len(test_images)}"
        )

    print("\nГотово.")


if __name__ == '__main__':
    if not os.path.exists(SOURCE_ROOT):
        raise "Скачайте датасет HaGRIDv2_dataset_512"

    split_dataset()
