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


from tqdm.auto import tqdm


@torch.no_grad()
def estimate_color_prior(train_loader, palette, device, stride=4):
    num_classes = palette.shape[0]
    counts = torch.zeros(num_classes, dtype=torch.float64)
    palette = palette.to(device=device, dtype=torch.float32)

    for _, ab in tqdm(train_loader, desc="Estimating color distribution"):
        ab = ab.to(device=device, dtype=torch.float32)

        #берем каждый четвертый пиксель по высоте и ширине для ускорения
        sampled = ab[:, :, ::stride, ::stride]

        #получаем пары координат ab для выбранных пикселей
        pixels = (sampled * 128.0).permute(0, 2, 3, 1).reshape(-1, 2)

        #находим ближайший цветовой класс для каждого пикселя
        distances = torch.cdist(pixels, palette)
        nearest = distances.argmin(dim=1)

        #подсчитываем количество пикселей каждого класса
        counts += torch.bincount(nearest, minlength=num_classes).cpu()


    prior = (counts / counts.sum()).float()

    return prior


def make_class_weights(prior, palette, lam=0.5, smoothing_sigma=5.0):
    prior = prior.float().cpu()
    palette = palette.float().cpu()

    #сглаживаем распределение цветов гауссовым ядром
    distances = torch.cdist(palette, palette)
    kernel = torch.exp(-(distances**2) / (2 * smoothing_sigma**2))
    kernel = kernel / kernel.sum(dim=1, keepdim=True)

    smooth_prior = prior @ kernel
    smooth_prior = smooth_prior / smooth_prior.sum()

    #смешиваем с равномерным распределением
    mixed_prior = (1 - lam) * smooth_prior + lam / len(prior)

    #редким классам назначаем большие веса
    class_weights = 1.0 / mixed_prior

    #нормализуем веса так, чтобы средний вес был равен 1
    class_weights = class_weights / (smooth_prior * class_weights).sum()

    return class_weights

def make_class_weights(prior, palette, lam=0.5, smoothing_sigma=5.0):
    prior = prior.float().cpu()
    palette = palette.float().cpu()

    #сглаживаем распределение цветов гауссовым ядром
    distances = torch.cdist(palette, palette)
    kernel = torch.exp(-(distances**2) / (2 * smoothing_sigma**2))
    kernel = kernel / kernel.sum(dim=1, keepdim=True)

    smooth_prior = prior @ kernel
    smooth_prior = smooth_prior / smooth_prior.sum()

    #смешиваем с равномерным распределением
    mixed_prior = (1 - lam) * smooth_prior + lam / len(prior)

    #редким классам назначаем большие веса
    class_weights = 1.0 / mixed_prior

    #нормализуем веса так, чтобы средний вес был равен 1
    class_weights = class_weights / (smooth_prior * class_weights).sum()

    return class_weights