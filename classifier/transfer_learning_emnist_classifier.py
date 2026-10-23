import time

import torch
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


from common.pytorch.ml_wrapper import ML_Wrapper
from examples.CNN.pytorch.cnn_model import ConvNet

base_path = "../../../"
n_epochs = 20
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
transfer_learning = False
datasets_path = "../datasets/"
transfer_learning = True

#print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.EMNIST(datasets_path, train=True, download=True,
        split='letters', #ma 26 klas może być 'balanced' ale ma 47 klas,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1751,), (0.3332,)) # Nowa normalizacja
        ])
    ),
    batch_size=batch_size_train, shuffle=True,**data_loader_kwargs )

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.EMNIST(datasets_path, train=False, download=True,
        split='letters', #ma 26 klas może być 'balanced' ale ma 47 klas,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1751,), (0.3332,)) # Nowa normalizacja
        ])
    ),
    batch_size=batch_size_test, shuffle=True, **data_loader_kwargs )