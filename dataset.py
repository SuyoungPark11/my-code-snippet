import torch
from torchvision.io import read_image
from typing import Callable, Union, Tuple
import pandas as pd
import os

class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        root_dir: str, 
        info_df: pd.DataFrame, # including labels & path 
    ):
        self.root_dir = root_dir
        self.image_labels = info_df['lable'].tolist()
        self.image_paths = info_df['image_path'].tolist()
        
    def __len__(self) -> int: # number of images 
        return len(self.image_labels)

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        img_path = os.path.join(self.root_dir, self.image_paths[index])
        image = read_image(img_path)
        label = self.image_labels[index]
        return image, label  