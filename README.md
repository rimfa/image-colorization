# Image Colorization with PyTorch

Проект по автоматической колоризации чёрно-белых изображений с использованием PyTorch. Исследуются архитектуры U-Net и различные подходы к улучшению качества колоризации.

## Задача

Модель получает канал яркости `L` из цветового пространства LAB и предсказывает цветовые каналы `a` и `b` для восстановления цветного изображения.

- **Датасет:** [PASCAL VOC 2012](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/#devkit).
- **Размер изображений:** 128 × 128 (Resize + CenterCrop).
- **Вход:** `L` — `[B, 1, 128, 128]`, нормализация `L / 100`.
- **Выход:** `ab` — `[B, 2, 128, 128]`, нормализация `ab / 128`.

## Baseline: U-Net

Реализована собственная U-Net с тремя уровнями downsampling, skip connections и `base_channels=32`, без BatchNorm.

**Обучение:** L1 Loss, AdamW (`lr=1e-3`), batch size 32. Первоначально модель обучалась 20 эпох, затем была дообучена ещё 20 эпох (всего 40).

## U-Net with Pretrained ResNet18 Encoder

В следующем эксперименте encoder U-Net был заменён на ResNet18 с предобученными весами ImageNet. Decoder реализован на основе блоков U-Net со skip connections.

На данном этапе веса encoder заморожены, обучается только decoder. Входной канал `L` дублируется до трёх каналов с последующей нормализацией ImageNet.

**Обучение:** L1 Loss, AdamW (`lr=1e-3`), batch size 32, 20 эпох.

## Результаты

Обучение проводилось в Google Colab с использованием GPU, метрики отслеживались в Weights & Biases.

Сравнение L1 Loss на одном validation-батче из 32 изображений:

| Модель | L1 Loss |
|---|---:|
| Grayscale baseline (`ab = 0`) | 0.06416 |
| U-Net — 20 эпох | 0.06259 |
| U-Net — 40 эпох | 0.06227 |
| **ResNet18-U-Net — 20 эпох** | **0.06049** |

Использование предобученного ResNet18 позволило снизить L1 Loss примерно на **2.9%** по сравнению с U-Net после 40 эпох.

Визуально новая модель стала точнее определять цветовые области, особенно на лицах людей. Однако проблема недостаточной насыщенности сохранилась: одежда и другие объекты с неоднозначными цветами часто приобретают нейтральные оттенки.

Анализ каналов `ab` также показал, что предсказанные абсолютные значения цветовых компонентов остаются значительно ниже исходных.

*Приведённые значения L1 рассчитаны на отдельном validation-батче, а не на всей validation-выборке.*

[U-Net baseline](notebooks/03_unet_baseline.ipynb) · [ResNet18-U-Net](notebooks/04_pretrained_resnet.ipynb) · [W&B](https://forge.coreweave.com/wandb/arzushkaB/image-colorization?nw=nwuserarzun9530)

## Структура проекта

```text
notebooks/
├── 00_data_eda.ipynb
├── 01_color_spaces.ipynb
├── 02_dataset_dataloader.ipynb
├── 03_unet_baseline.ipynb
└── 04_pretrained_resnet.ipynb

src/image_colorization/
├── color_utils.py
├── dataset.py
├── dataloaders.py
├── training.py
├── unet.py
└── resnet_unet.py
```

Установка локального пакета:

```bash
pip install -e .
```

Датасет и checkpoints не включены в репозиторий.

## Дальнейшие эксперименты

- Scheduler для более стабильного обучения.
- Fine-tuning предобученного ResNet18 encoder.
- Альтернативные функции потерь для улучшения насыщенности цветов.
- Сравнение качества на всей validation-выборке.