import numpy as np
import torch

def load_palette(path):
    palette = torch.from_numpy(np.load(path)).float()
    return palette

def soft_encode_ab(ab, k, sigma, palette):
     B, _, H, W = ab.shape

     pixels = (ab * 128.0).permute(0, 2, 3, 1).reshape(-1, 2)
     #переходим из ab представления в представление в виде вектора расстояний до всех 313 цветов палитры
     distances = torch.cdist(pixels, palette)
     #выбираем 5 самых близких цветов
     dist, idx = torch.topk(distances, k=k, largest=False, dim=1)
     #каждой выбранной координате даем свой вес. чем ближе объект к исходному ab тем больше вес
     weights = torch.exp(-((dist**2)/(2*sigma**2)))
     #нормализуем веса (что бы сумма была 1)
     norm_weights = weights/(weights.sum(dim=1, keepdim=True))

     idx = idx.reshape(B, H, W, k).permute(0, 3, 1, 2)
     norm_weights = norm_weights.reshape(B, H, W, k).permute(0, 3, 1, 2)

     return idx, norm_weights
