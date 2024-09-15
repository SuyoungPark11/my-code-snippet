import torch
import torch.nn as nn
from torchvision.io import read_image
from typing import Callable, Union, Tuple
import pandas as pd
import os

class ImageDataset(torch.utils.data.Dataset):
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


class Loss(nn.Module):
    def __init__(
            self,
            loss_fn: str
    ):
        super(Loss, self).__init__()
        loss_dict = {
            "L1" : nn.L1Loss,
            "MSE" : nn.MSELoss,
            "CE" : nn.CrossEntropyLoss,
            "BCE" : nn.BCELoss
        }
        self.loss_fn = loss_dict[loss_fn]

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
    
        return self.loss_fn(outputs, targets)
