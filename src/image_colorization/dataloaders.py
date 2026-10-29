from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import ColorizationDataset


def create_dataloaders(images_dir,train_path,val_path,
                       batch_size=32,num_workers=2,image_size=128):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size)
    ])

    train_dataset = ColorizationDataset(images_dir=Path(images_dir), split_path=Path(train_path),
                                        transform=transform)

    val_dataset = ColorizationDataset(images_dir=Path(images_dir), split_path=Path(val_path),
                                      transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    return train_loader, val_loader