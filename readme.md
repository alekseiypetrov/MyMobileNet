# Распознавание жестов с MobileNetV3 и CoreML

Проект по распознаванию жестов рук с использованием нейронной сети MobileNetV3, PyTorch и CoreML.

## Используемые технологии

- Python
- PyTorch
- torchvision
- MobileNetV3
- CoreMLTools
- scikit-learn

## Используемый датасет

Проект использует датасет HaGRID.

Поддерживаемые классы жестов:

- call
- like
- no_gesture
- ok
- palm
- peace

## Подготовка датасета

### 1. Разделение датасета

```bash
python split_dataset.py
```

Скрипт:

-   выбирает изображения нужных классов;
-   случайно перемешивает их;
-   делит на train / val / test.
    

### 2. Предобработка изображений

```bash
python filter_preprocess_dataset.py
```

Скрипт:

-   считывает bbox-аннотации;
-   выделяет область руки;
-   изменяет размер изображения до 224x224.

## Обучение модели

```bash
python setup_train_test_val.py
```

Во время обучения:

-   используется трансферное обучение;
-   замораживаются feature layers MobileNetV3;
-   обучается только classifier head;
-   применяется ранняя остановка;
-   сохраняется лучшая модель.

## Метрики

После обучения выводятся:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Confusion Matrix

## Конвертация в CoreML

После обучения модель автоматически конвертируется в CoreML:

```text
GestureClassifier.mlpackage
```

## Важно

CoreML conversion рекомендуется выполнять на macOS.

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Автор

Магистерская диссертация по распознаванию жестов рук для мобильного приложения iOS.
