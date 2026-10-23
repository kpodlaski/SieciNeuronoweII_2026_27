import os
import time

import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
from sklearn.metrics._classification import confusion_matrix

class MLWrapper():
    def __init__(self, network, optimizer, base_path, out_path, device, noise_factor=0.2):
        self.network=network
        self.noise_factor=noise_factor
        self.network.to(device)
        self.optimizer = optimizer
        self.device = device
        self.base_path = base_path
        self.out_path = out_path
        self.train_losses = []
        self.val_losses = []

    def train(self, epoch, data_loader, val_loader = None):
        t0 = time.time()
        self.network.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(self.device)
            target = target.to(self.device)
            self.optimizer.zero_grad()
            output = self.network(data)
            loss = F.nll_loss(output, target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        t0 = time.time() - t0
        print('Train Epoch: {} \tLoss: {:.6f}, epoch time(s):{:.4f}'.format(
            epoch, train_loss, t0))
        self.train_losses.append(train_loss)
        if (val_loader):
            self.test(val_loader, val_test=True)

    def test(self, data_loader, val_test=False, create_confusion=False):
        if self.optimizer is None:
            raise Exception("No optimizer is set")
        self.network.eval()
        test_loss = 0
        correct = 0
        if create_confusion:
            inialized = None
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.network(data)
                test_loss += F.nll_loss(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
                if create_confusion:
                    if inialized == None:
                        results = pred.cpu().detach().numpy().reshape(-1)
                        expected = target.cpu().detach().numpy().reshape(-1)
                        inialized = True
                    else:
                        results = np.concatenate((results, pred.cpu().detach().numpy().reshape(-1)), axis=None)
                        expected = np.concatenate((expected, target.cpu().detach().numpy().reshape(-1)), axis=None)
        if (val_test):
            self.val_losses.append(test_loss)
        else:
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(data_loader.dataset),
                100. * correct / len(data_loader.dataset)))
        if create_confusion:
            cf_matrix = confusion_matrix(expected, results)
            return cf_matrix

    def add_noise(self, inputs):
        noisy = inputs + torch.randn_like(inputs) * self.noise_factor
        noisy = torch.clip(noisy, 0., 1.)
        return noisy

    def save_model(self, fname):
        torch.save(self.network.state_dict(), f'{self.base_path}/{self.out_path}/{fname}_{self.noise_factor}_model.pth')

    @staticmethod
    def get_model_path(base_path, sub_dir, file_name ):
        return f"{base_path}/{sub_dir}/{file_name}"

    @staticmethod
    def load_model(base_path, sub_dir, file_name, model, optimizer=None, device = None ):
        file_path = MLWrapper.get_model_path(base_path,sub_dir, file_name)
        model.load_state_dict(torch.load(file_path))
        return MLWrapper(model, optimizer,base_path,sub_dir,device)


    def training_plot(self):
        fig = plt.figure()
        plt.plot([*range(len(self.train_losses))], self.train_losses, color='blue')
        if (len(self.val_losses))>0:
            plt.plot([*range(len(self.val_losses))], self.val_losses, color='red')
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
        plt.xlabel('epoch')
        plt.ylabel('negative log likelihood loss')
        fig.savefig(f"{self.base_path}/{self.out_path}/train.png")
        plt.close()

    def summary(self, size):
        x = torch.Tensor(size).to(self.device)
        summary(self.network, size)

    def predict(self, x):
        return self.network(x)

