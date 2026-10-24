import numpy as np
from skimage.color import rgb2lab

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
