import torch # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torchvision.io import read_image # type: ignore
from typing import Union, Tuple, Callable
import pandas as pd # type: ignore
import os

class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root_dir, 
        info_df : pd.DataFrame, # including labels & path
        transform : Callable,
        is_inference : bool = False
    ):
        self.root_dir = root_dir
        self.image_paths = info_df['image_path'].tolist()
        self.transform = transform
        self.is_inference = is_inference

        if not self.is_inference:
            self.image_labels = info_df['lable'].tolist()
        
    def __len__(self) -> int: # number of images 
        return len(self.image_labels)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = read_image(img_path)
        
        if self.is_inference:
            return image
        else:
            label = self.image_labels[index]
            return image, label  

train_dataset = ImageDataset(root_dir="train_data_root")
test_dataset = ImageDataset(root_dir="test_data_root")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    drop_last=True
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    drop_last=True
)





