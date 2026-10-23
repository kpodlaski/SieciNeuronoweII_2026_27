import os
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torchsummary import summary

from autoencoder.AutoEncoder import AutoencoderNet

class AutoEncoderWrapper():
    def __init__(self, network, optimizer, base_path, device, noise_factor=0.2):
        self.network=network
        self.noise_factor=noise_factor
        self.network.to(device)
        self.optimizer = optimizer
        self.device = device
        self.base_path = base_path
        self.train_losses = []
        self.val_losses = []

    def train(self, epoch, data_loader, val_loader = None):
        self.network.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data_plus_noise = self.add_noise(data)
            data = data.to(self.device)
            data_plus_noise = data_plus_noise.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data_plus_noise)
            loss = torch.nn.MSELoss()(output, data)
            train_loss+= loss.item()
            loss.backward()
            self.optimizer.step()
        train_loss = train_loss/len(data_loader.dataset)
        print('Train Epoch: {} \tLoss: {:.6f}'.format(
            epoch, train_loss))
        self.train_losses.append(train_loss)

    def add_noise(self, inputs):
        noisy = inputs + torch.randn_like(inputs) * self.noise_factor
        noisy = torch.clip(noisy, 0., 1.)
        return noisy

    def save_model(self, fname):
        torch.save(self.network.state_dict(), self.base_path + '/out/{}_{}_c{}_model.pth'.format(fname, self.noise_factor, self.network.code_dim))

    @staticmethod
    def get_model_path(base_path, file_name ):
        return base_path + "/out/" + file_name

    @staticmethod
    def load_model(base_path, file_name, model, optimizer=None, device = None ):
        file_path = AutoEncoderWrapper.get_model_path(base_path,file_name)
        model.load_state_dict(torch.load(file_path))
        return AutoEncoderWrapper(model, optimizer,base_path,device)


    def training_plot(self):
        fig = plt.figure()
        plt.plot([*range(len(self.train_losses))], self.train_losses, color='blue')
        if (len(self.val_losses))>0:
            plt.plot([*range(len(self.val_losses))], self.val_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('negative log likelihood loss')
        fig.savefig("../out/autoencoder/train.png")
        plt.close()

    def summary(self, size):
        x = torch.Tensor(size).to(self.device)
        summary(self.network, size)

    def encode(self, x):
        return self.network.encode(x)

    def decode(self, x):
        return self.network.decode(x)
