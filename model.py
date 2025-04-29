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


# Dataset Sınıfı
class ExternalDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = A.Compose([
            A.Resize(240, 240),
            A.Normalize(mean=[0.485], std=[0.229]),
            ToTensorV2()
        ])

    def __len__(self): return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = np.stack([img]*3, axis=-1)
        img = self.transform(image=img)['image']
        label = self.df.iloc[idx]['label'] if 'label' in self.df.columns else -1
        return img, torch.tensor([label], dtype=torch.float32)




# Grad-CAM Sınıfı
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def get_cam(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        output.backward(retain_graph=True)
        
        weights = F.adaptive_avg_pool2d(self.gradients, 1)
        cam = torch.mul(self.activations, weights).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, (240,240), mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.squeeze().cpu().numpy()        



# Model Yükleme
def load_model():
    model = EfficientNet.from_pretrained('efficientnet-b1')
    model._fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(1280, 1))
    model.load_state_dict(torch.load('D:/TEKNOFEST/BTmakinesiEFFNET-V2S/best_model.pth', map_location=DEVICE))
    return model.to(DEVICE)


# Görselleştirme Fonksiyonu
def visualize_gradcam(image, cam, pred, true_label=None, save_path=None):
    image = image.squeeze().permute(1,2,0).numpy()[:,:,0]
    image = (image * 0.229 + 0.485) * 255  # Unnormalize
    image = np.uint8(np.clip(image, 0, 255))
    
    cam = cv2.resize(cam, (240,240))
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    combined = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.5, heatmap, 0.5, 0)
    
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title("Original")
    
    plt.subplot(122)
    plt.imshow(combined)
    title = f"Pred: {'Stroke' if pred>0.5 else 'Normal'}"
    if true_label is not None:
        title += f" | True: {'Stroke' if true_label==1 else 'Normal'}"
    plt.title(title)
    
    if save_path:
        plt.imsave(save_path, combined)
    plt.close()
