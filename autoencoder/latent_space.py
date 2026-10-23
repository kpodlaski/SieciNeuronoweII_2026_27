import os

import numpy as np
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torchsummary import summary

from autoencoder.AutoEncoder import AutoencoderNet
from autoencoder.AutoEncoderWrapper import AutoEncoderWrapper

def mean_latent_for_class(model, data_loader, class_id, device):
    _size = 0
    latent = torch.zeros((1, code_dim), device=device)
    ml.network.to(device)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            mask = (target == class_id)
            if not mask.any():
                continue
            filtered_data = data[mask].to(device)
            encoded = model.encode(filtered_data)
            latent += encoded.sum(dim=0, keepdim=True)
            _size += filtered_data.size(0);
        latent = latent / _size
    return latent

print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
base_path = ".."
datasets_path = "../datasets/"
batch_size_train = 24
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(datasets_path, train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_train, shuffle=True, **data_loader_kwargs)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(datasets_path, train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
    batch_size=batch_size_test, shuffle=True, **data_loader_kwargs)
code_dim = 12

model_file_name = "autoencoder_0.2_c12_model.pth"
code_dim = 12
network = AutoencoderNet(code_dim)
optimizer = torch.optim.Adam(network.parameters())

ml = AutoEncoderWrapper(network, optimizer, base_path, device)
ml.summary((1,28,28))

if os.path.exists(AutoEncoderWrapper.get_model_path(base_path,model_file_name)):
    print("Loading model from file:", model_file_name)
    ml = AutoEncoderWrapper.load_model(base_path, model_file_name, network, optimizer=optimizer, device=device)
else:
    print("Nie ma wag dla sieci")
    print("Oczekiwano pliku: ", AutoEncoderWrapper.get_model_path(base_path,model_file_name))

class_id = 3
_size = 0
latent = torch.zeros((1,code_dim), device=device)
ml.network.to(device)
with torch.no_grad():
    latent = mean_latent_for_class(ml, train_loader, class_id, device)
    img = ml.decode(latent)
    img = img.cpu()
    fig = plt.figure()
    plt.imshow(img[0][0], cmap='gray', interpolation='none')
    plt.title("from latent: ")
    plt.tight_layout()
    fig.savefig("../out/autoencoder/latent_train_class_{}.png".format(class_id))
    plt.close()

for class2_id in [1,3]:
    with torch.no_grad():
        latent2 = mean_latent_for_class(ml, test_loader, class2_id, device)
        img = ml.decode(latent2)
        img = img.cpu()
        fig = plt.figure()
        plt.imshow(img[0][0], cmap='gray', interpolation='none')
        plt.title("from latent: ")
        plt.tight_layout()
        fig.savefig("../out/autoencoder/latent_test_class_{}.png".format(class2_id))
        plt.close()
        mse_dist = torch.mean((latent - latent2) ** 2)
        print("MSE between: train {} a test {}: {} ".format(class_id, class2_id, mse_dist))
        dot_product = torch.sum(latent * latent2)
        norm_a = torch.norm(latent)
        norm_b = torch.norm(latent2)
        cos_sim = dot_product / (norm_a * norm_b)
        cos_dist = 1.0 - cos_sim
        print("Cosine dist between: train {} a test {}: {} ".format(class_id, class2_id, cos_dist))