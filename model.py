# Gerekli Kütüphaneler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# Yapılandırma
EXTERNAL_DATA_PATH = "./external_data"
GRADCAM_OUTPUT = "./gradcam_output"
CLASSES = {"normal": 0, "stroke": 1}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Harici Veri için DataFrame
def create_external_df():
    if os.path.exists(os.path.join(EXTERNAL_DATA_PATH, "labels.csv")):
        df = pd.read_csv(os.path.join(EXTERNAL_DATA_PATH, "labels.csv"))
        return df
    else:
        data = []
        for class_name in CLASSES:
            class_dir = os.path.join(EXTERNAL_DATA_PATH, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    data.append({
                        "path": os.path.join(class_dir, img_name),
                        "label": CLASSES[class_name]
                    })
        if not data:  # Sadece images klasörü
            images_dir = os.path.join(EXTERNAL_DATA_PATH, "images")
            if os.path.exists(images_dir):
                for img_name in os.listdir(images_dir):
                    data.append({
                        "path": os.path.join(images_dir, img_name),
                        "label": -1  # Bilinmeyen etiket
                    })
        return pd.DataFrame(data)