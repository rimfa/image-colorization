import torch
import torch.nn as nn
import torch.nn.functional as F

from .color_quantization import soft_encode_ab

class SoftColorLoss(nn.Module):
    def __init__(self, palette, k=5, sigma=5.0):
        super().__init__()

        self.register_buffer("palette", palette.float())
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
        #считаем взвешенную кросс-энтропию
        loss = -(weights * selected_log_probs).sum(dim=1).mean()

        return loss