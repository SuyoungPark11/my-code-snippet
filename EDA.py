import os
import cv2  # type: ignore
import torch  # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
from PIL import Image # type: ignore

bins = torch.linspace(0, 256, 257)
sum_hist_L = torch.zeros(256)
sum_hist_R = torch.zeros(256)
sum_hist_G = torch.zeros(256)
sum_hist_B = torch.zeros(256)
image_L_count = 0
image_RGB_count = 0
traindata_info = pd.DataFrame('data source')


for i in range(len(traindata_info)):
    image_path = traindata_info.iloc[i, :]['image_path']
    image = Image.open(image_path)
    mode = image.mode
    if mode == 'L':
        image_tensor = torch.tensor(np.array(image)).float()
        hist, _ = torch.histogram(image_tensor, bins=bins)
        sum_hist_L += hist
        image_L_count += 1
    elif mode == 'RGB': 
        image_tensor = torch.tensor(np.array(image)).float()
        hist_r, _ = torch.histogram(image_tensor[:,:,0], bins=bins)
        hist_g, _ = torch.histogram(image_tensor[:,:,1], bins=bins)
        hist_b, _ = torch.histogram(image_tensor[:,:,2], bins=bins)
        sum_hist_R += hist_r
        sum_hist_G += hist_g
        sum_hist_B += hist_b
        image_RGB_count += 1