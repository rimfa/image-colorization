import numpy as np
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
from .color_utils import prepare_lab
class ColorizationDataset(Dataset):
    def __init__(self, images_dir, split_path, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.filenames = split_path.read_text().splitlines()

        self.transform = transform
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        file_name = self.filenames[index]
        with Image.open(os.path.join(self.images_dir, file_name)) as img:
            rgb_img = img.convert("RGB")

        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
        rgb_img = np.array(rgb_img)

        L_norm, ab_norm = prepare_lab(rgb_img)
        L_norm = torch.tensor(L_norm[np.newaxis, :, :], dtype=torch.float32)
        ab_norm = torch.tensor(np.transpose(ab_norm, (2, 0, 1)), dtype=torch.float32)

        return L_norm, ab_norm