# Image Colorization with PyTorch

Проект посвящён автоматической колоризации чёрно-белых изображений с использованием PyTorch. В рамках работы исследуются архитектуры на основе U-Net, различные подходы к обучению и применение предобученных моделей для улучшения качества колоризации.

## Задача

Модель получает канал lightness **L** из цветового пространства CIELAB и предсказывает два цветовых канала **a** и **b**. Вместе с исходным L они позволяют восстановить цветное изображение.

- **Данные:** [PASCAL VOC 2012](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/#devkit).
- **Размер изображений:** 128 × 128 (Resize + CenterCrop).
- **Вход:** `L` — `[B, 1, 128, 128]`, нормализация `L / 100`.
- **Выход:** `ab` — `[B, 2, 128, 128]`, нормализация `a / 128`, `b / 128`.

## Текущий baseline

Custom **U-Net** на PyTorch: блоки `DoubleConv`, `Down`, `Up`, bottleneck и skip connections с конкатенацией. Три уровня downsampling, `base_channels=32`, без BatchNorm.

**Обучение:** L1 Loss, AdamW (`lr=1e-3`), batch size 32, 20 эпох. Эксперимент выполнен в Google Colab с GPU; метрики отслеживаются в Weights & Biases (W&B). Сохраняются `best` и `last` checkpoints на Google Drive.

## Эксперименты

| Эксперимент | Архитектура / изменения | Эпохи | Train L1 | Val L1 |
|---|---|---:|---:|---:|
| Baseline v1 | U-Net с нуля | 20 | 0.06448 | 0.06289 |

*Показаны метрики последней, 20-й эпохи на нормализованных `ab`-каналах; это не test-метрики и не обязательно лучший validation loss за весь запуск.*

[Результаты baseline в W&B](https://wandb.ai/arzushkaB/image-colorization/runs/ngviphfh) · [Ноутбук обучения](notebooks/03_unet_baseline.ipynb)

## Структура проекта

```text
notebooks/
├── 00_data_eda.ipynb           # Исследование данных и train/val/test splits
├── 01_color_spaces.ipynb       # RGB ↔ LAB, нормализация
├── 02_dataset_dataloader.ipynb # Dataset и DataLoader
└── 03_unet_baseline.ipynb      # U-Net, обучение, W&B и checkpoints

src/image_colorization/
├── color_utils.py
├── dataset.py
└── unet.py
```

Реализации вынесены в Python-пакет `src/`. Для работы с кодом в подготовленном окружении используется editable installation:

```bash
pip install -e .
```

Датасет и checkpoints не включены в Git-репозиторий.

## Дальнейшие эксперименты

Планируется сравнить scheduler, аугментации, BatchNorm и U-Net с **pretrained ResNet18 encoder**. Помимо L1 Loss будут оцениваться визуальные результаты колоризации; итоговая оценка на test split пока не проводилась.