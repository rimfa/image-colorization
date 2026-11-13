# Image Colorization with PyTorch

A PyTorch project for automatic image colorization. Two architectures are implemented and compared: a U-Net trained from scratch and a U-Net with a pretrained ResNet18 encoder.

## Task

Image colorization is formulated as predicting the **a and b color channels** from the **L (lightness) channel** in the CIE Lab color space.

- **Dataset:** [PASCAL VOC 2012](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/#devkit)
- **Image size:** 128 × 128
- **Input:** L channel — `(B, 1, 128, 128)`
- **Output:** ab channels — `(B, 2, 128, 128)`
- **Data split:** 70% training / 15% validation / 15% testing

## Models

### U-Net

A custom U-Net implementation with three downsampling stages and skip connections. The model was trained from scratch using L1 Loss and the AdamW optimizer for a total of 40 epochs.

### ResNet18-U-Net

A U-Net architecture with a ResNet18 encoder pretrained on ImageNet. Three training configurations were evaluated:

- **Frozen Encoder:** Training the decoder while keeping the encoder weights frozen.
- **ReduceLROnPlateau:** Using a learning rate scheduler to adjust the learning rate during training.
- **Fine-tuning:** Unfreezing and fine-tuning the last encoder block (`layer4`).

All experiments used L1 Loss for regression-based color prediction.

## Results

L1 Loss comparison on a single validation batch of 32 images:

| Model | L1 Loss |
|---|---:|
| Grayscale Baseline (`ab = 0`) | 0.06416 |
| U-Net (40 epochs) | 0.06227 |
| ResNet18-U-Net (Frozen Encoder) | **0.06049** |
| ResNet18-U-Net + Scheduler | 0.06084 |
| ResNet18-U-Net (Fine-tuning) | 0.06172 |

The pretrained ResNet18 encoder reduced the prediction error compared to the original U-Net.

Evaluation of the frozen encoder model on the full validation set (2,569 images) achieved a **mean L1 loss of 0.0597**, outperforming the grayscale baseline on **71.4% of images**.

Despite the improvement in L1 Loss, the predicted colors often remained desaturated.

Models were trained in Google Colab using GPU acceleration. Training histories and experiment metrics are available in [Weights & Biases](https://wandb.ai/arzushkaB/image-colorization).

## Project Structure

```text
notebooks/
├── 00_data_exploration.ipynb
├── 01_unet_baseline.ipynb
└── 02_resnet_unet.ipynb

src/image_colorization/
├── dataset.py
├── dataloaders.py
├── color_utils.py
├── unet.py
├── resnet_unet.py
└── training.py
```

The notebooks contain dataset exploration, training results, model evaluation, and colorization examples.

## Usage

Clone the repository and install the project:

```bash
git clone https://github.com/rimfa/image-colorization.git
cd image-colorization
pip install -e .
```

Download the PASCAL VOC 2012 dataset and place the images in:

```text
VOCdevkit/VOC2012/JPEGImages/
```

Place the pretrained model checkpoints in:

```text
checkpoints/
```

The dataset and model checkpoints are not included in the repository. The notebooks contain saved experiment results that can be viewed without retraining the models.