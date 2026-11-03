
import torch
import torch.nn as nn
import torch.nn.functional as F

from .color_quantization import soft_encode_ab


class SoftColorLoss(nn.Module):
    def __init__(self, palette, class_weights=None, k=5, sigma=5.0):
        super().__init__()

        self.register_buffer("palette", palette.float())

        if class_weights is None:
            class_weights = torch.ones(palette.shape[0])

        self.register_buffer("class_weights", class_weights.float())

        self.k = k
        self.sigma = sigma

    def forward(self, logits, ab):
        palette = self.palette.to(device=ab.device, dtype=ab.dtype)

        #получаем индексы ближайших классов и их веса
        idx, weights = soft_encode_ab(ab, self.k, self.sigma, palette)

        #преобразуем логиты в логарифмы вероятностей
        log_probs = F.log_softmax(logits, dim=1)

        #выбираем вероятности только k ближайших классов
        selected_log_probs = torch.gather(log_probs, dim=1, index=idx)

        #считаем soft cross-entropy для каждого пикселя
        loss = -(weights * selected_log_probs).sum(dim=1)

        #получаем вес ближайшего цветового класса каждого пикселя
        pixel_weights = self.class_weights[idx[:, 0, :, :]]

        #балансируем ошибку и усредняем
        loss = (loss * pixel_weights).mean()

        return loss
