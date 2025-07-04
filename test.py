import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math
from tqdm.notebook import trange, tqdm
import random

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
#from torch.distributions import Categorical

import torchvision
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image
from skimage.color import rgb2lab, lab2rgb

from pathlib import Path

import time

import wandb

torch.backends.cuda.matmul.allow_tf32 = True

data_set_root='/kaggle/input/coco-2017-dataset/coco2017'
train_set ='train2017'
validation_set ='val2017'
test_set = 'test2017'

train_path = os.path.join(data_set_root, train_set)

val_path = os.path.join(data_set_root, validation_set)

test_path = os.path.join(data_set_root, test_set)



train_image_path = list(Path(train_path).rglob("*.*"))
val_image_path = list(Path(val_path).rglob("*.*"))
test_image_path = list(Path(test_path).rglob("*.*"))

print(len(train_image_path), len(val_image_path), len(test_image_path))



def setup(rank, world_size):
    """ Initialize process group for DDP """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) # Ensure each process uses the correct GPU
    device = torch.device(f'cuda:{rank}')

    return device

def cleanup():
    """ Destroy process group when training is complete """
    dist.destroy_process_group()

image_size = 224

batch_size = 64

class ColorizationDataset(Dataset):
    def __init__(self, paths, Size=(224, 224), transform=None):
        self.paths = paths
        self.height, self.width = Size
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.height, self.width), Image.BICUBIC)
        img = np.array(img)  # Convert PIL to NumPy (Albumentations requires NumPy)

        # Apply Albumentations transform if provided
        if self.transform:
            transformed = self.transform(image=img)
            img = transformed["image"]

        # If img is in (C, H, W) format, convert to (H, W, C)
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))  # Convert (C, H, W) → (H, W, C)

        # Convert RGB → LAB
        img_lab = rgb2lab(img).astype("float32")  # (H, W, 3)

        # Extract L and ab channels
        L = img_lab[:, :, 0] 
        ab = img_lab[:, :, 1:]

        # Convert to PyTorch tensors
        L = torch.tensor(L, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        ab = torch.tensor(ab, dtype=torch.float32).permute(2, 0, 1)  # (2, H, W)

        return {'L': L, 'ab': ab}
    
transform = A.Compose([
    A.HorizontalFlip(p=0.4),
    A.VerticalFlip(p=0.4),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma (gamma_limit=(70, 130), p=0.2),
    ToTensorV2(),
])

def make_dataloaders(rank, world_size, batch_size=16, n_workers=4, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader


class BaseColor(nn.Module):
	def __init__(self):
		super(BaseColor, self).__init__()

		self.l_cent = 50.
		self.l_norm = 100.
		self.ab_norm = 110.

	def normalize_l(self, in_l):
		return (in_l-self.l_cent)/self.l_norm   # Normalize L to [-0.5, 0.5]

	def unnormalize_l(self, in_l):
		return in_l*self.l_norm + self.l_cent

	def normalize_ab(self, in_ab):
		return in_ab/self.ab_norm    # Normalize ab to [-1, 1]

	def unnormalize_ab(self, in_ab):
		return in_ab*self.ab_norm
      

class ColorizationNet(BaseColor):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(ColorizationNet, self).__init__()

        model1=[nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]

        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]

        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]

        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        model8=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),]
        model8+=[nn.ReLU(True),]

        model8+=[nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, input_l):
        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = self.model_out(self.softmax(conv8_3))

        return self.unnormalize_ab(self.upsample4(out_reg))
    
def create_model(rank):
    model = ColorizationNet().to(rank)  # Move model to GPU
    model = DDP(model, device_ids=[rank])  # Wrap with DDP
    return model


learning_rate = 1e-3

epochs = 200

model_path = '/kaggle/working/model.pth'


wandb.login(
    key = "d9d14819dddd8a35a353b5c0b087e0f60d717140",
)

# Early stopping parameters
patience = 15  # Number of epochs to wait for improvement
counter = 0   # Counter to track the number of epochs without improvement
best_val_loss = float('inf')  # Initialize the best validation loss
best_epoch = 0  # Track the epoch when the best model was found


from torch.multiprocessing import Manager

manager = Manager()
shared_dict = manager.dict({"learning_rate": learning_rate, 
                           "epochs": epochs,
                           "model_path" : model_path,
                           })  # Shared across ranks


# Training function
def train(rank, world_size, args):
    
    device = setup(rank, world_size)
    
    # WandB init (only rank 0 logs)
    if rank == 0:
        wandb.init(project=PROJECT,
                    resume=RESUME,
                    name="init_colorize",
                    config={
                        "learning_rate": learning_rate,
                        "epochs": epochs,
                        "batch_size": batch_size,
                    },
            )
    
    # Dataset and DataLoader
    train_loader = make_dataloaders(rank, world_size, batch_size = batch_size, paths=train_image_path, transform = transform)
    
    # Model setup
    model = create_model(rank=rank)
    
    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1, verbose=True)
    early_stopping = EarlyStopping(patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(200):
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0
        y_true, y_pred = [], []
        
        for batch in train_loader:
            L, labels = batch['L'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(L).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.detach().cpu().numpy())
        
        train_loss = running_loss / len(train_loader)
        train_auc = roc_auc_score(y_true, y_pred)
        
        # Validation
        val_loss, val_auc = validate(model, criterion, device)
        
        # Scheduler and Early Stopping
        scheduler.step(val_loss)
        if early_stopping(val_loss):
            print("Early stopping triggered!")
            break
        
        # Logging
        if rank == 0:
            wandb.log({"train_loss": train_loss, "train_auc": train_auc, "val_loss": val_loss, "val_auc": val_auc})
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), '/kaggle/working/model.pth')
                print(f"Saved new best model with val_loss: {val_loss:.4f}")
    
    cleanup()


def validate(model, criterion, device):
    model.eval()
    val_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for _ in range(10):  # Placeholder for actual validation loader
            L = torch.rand(32, 3, 224, 224).to(device)
            labels = torch.randint(0, 2, (32,)).float().to(device)
            outputs = model(L).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    val_loss /= 10
    val_auc = roc_auc_score(y_true, y_pred)
    return val_loss, val_auc