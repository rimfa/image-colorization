import numpy as np
from skimage.color import rgb2lab
import torch
import torch.nn.functional as F

def normalize_lab(image):
    L = image[:, :, 0]
    a = image[:, :, 1]
    b = image[:, :, 2]
    
    L_norm = L/100
    ab_norm = np.stack([a/128, b/128], axis=2)

    return L_norm, ab_norm
def denormalize_lab(L_norm, ab_norm):
    L_denorm= L_norm*100
    a_denorm = ab_norm[:, :, 0]*128
    b_denorm= ab_norm[:, :, 1]*128

    return np.stack([L_denorm, a_denorm, b_denorm], axis=2)

def prepare_lab(image):
    lab_image = rgb2lab(image)
    L_norm, ab_norm = normalize_lab(lab_image)

    return L_norm, ab_norm


def annealed_mean_decode(logits, palette, temperature=0.38):
    #переносим палитру на то же устройство, что и предсказания
    palette = palette.to(device=logits.device, dtype=logits.dtype)
    #преобразуем логиты в вероятности с температурой
    probs = F.softmax(logits / temperature, dim=1)
    #получаем координаты ab как взвешенное среднее цветов палитры
    ab = torch.einsum("bqhw,qc->bchw", probs, palette)
    #нормализуем ab аналогично нашему датасету
    ab = ab / 128.0

    return ab