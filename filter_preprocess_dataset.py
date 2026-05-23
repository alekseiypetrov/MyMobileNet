import json
from pathlib import Path
from PIL import Image

# Пути к данным
DATASET_ROOT = Path(r".\MyDataset")
OUTPUT_ROOT = Path(r".\ProcessedDataset")
ANNOTATIONS_ROOT = Path(r".\annotations")

# Константы
IMAGE_SIZE = 224
PADDING = 0.15


# Загрузка всех JSON-аннотаций
def load_annotations():
    annotations = {}
    json_files = list(ANNOTATIONS_ROOT.rglob("*.json"))
    print(f"Найдено JSON файлов: {len(json_files)}")
    for json_path in json_files:
        print(f"Loading: {json_path.name}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            annotations.update(data)
    print(f"Всего аннотаций: {len(annotations)}")
    return annotations


# Обработка датасета
def process_dataset(annotations):
    splits = ["train", "val", "test"]
    for split in splits:
        split_dir = DATASET_ROOT / split
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            output_class_dir = OUTPUT_ROOT / split / class_dir.name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            images = list(class_dir.glob("*.jpg"))
            print(f"\n[{split}] {class_dir.name}: {len(images)} images")

            for image_path in images:
                try:
                    # Получение ImageID
                    image_id = image_path.stem
                    if image_id not in annotations:
                        print(f"[WARNING] Нет аннотации: {image_id}")
                        continue
                    annotation = annotations[image_id]

                    # bbox (выделение области нужной руки с жестом)
                    if not annotation["bboxes"]:
                        print(f"[WARNING] Нет bbox: {image_id}")
                        continue
                    bbox = None
                    for current_bbox, label in zip(annotation["bboxes"], annotation["labels"]):
                        if label == class_dir.name:
                            bbox = current_bbox
                            break
                    if bbox is None:
                        print(f"[WARNING] Нет bbox для класса {class_dir.name}: {image_id}")
                        continue
                    x, y, width, height = bbox

                    # Загрузка изображения
                    image = Image.open(image_path).convert("RGB")
                    img_w, img_h = image.size

                    # Вычисление координат относительно изображения
                    left = int(x * img_w)
                    top = int(y * img_h)
                    right = int((x + width) * img_w)
                    bottom = int((y + height) * img_h)

                    left = max(0, left)
                    top = max(0, top)
                    right = min(img_w, right)
                    bottom = min(img_h, bottom)

                    # Обрезка изображения
                    if right <= left or bottom <= top:
                        print(f"[WARNING] Invalid bbox: {image_id}")
                        continue
                    cropped = image.crop((left, top, right, bottom))

                    # Уменьшение изображения до 224х224
                    resized = cropped.resize((IMAGE_SIZE, IMAGE_SIZE))

                    # Сохранение
                    save_path = (output_class_dir / image_path.name)
                    resized.save(save_path)

                except Exception as e:
                    print(f"[ERROR] {image_path.name}: {e}")
    print("\nPreprocessing completed.")


if __name__ == '__main__':
    annot = load_annotations()
    process_dataset(annot)
