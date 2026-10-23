import os
import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torchsummary import summary

from autoencoder.AutoEncoder import AutoencoderNet
from autoencoder.AutoEncoderWrapper import AutoEncoderWrapper

print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
base_path = ".."
datasets_path = "../datasets/"
n_epochs = 30
batch_size_train = 24
batch_size_test = 1000
learning_rate = 0.001
momentum = 0.5
log_interval = 10
noise_factor = 0.2

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
network = AutoencoderNet(code_dim)
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-05)

model_file_name = "_autoencoder_0.3_c24_model.pth"


ml = AutoEncoderWrapper(network, optimizer, base_path, device)
ml.summary((1,28,28))

if os.path.exists(AutoEncoderWrapper.get_model_path(base_path,model_file_name)):
    print("Loading model from file:", model_file_name)
    ml = AutoEncoderWrapper.load_model(base_path, model_file_name, network, optimizer=optimizer, device=device)
else:
    print("Start training")
    for epoch in range(1, n_epochs + 1):
      ml.train(epoch, train_loader, test_loader)
    ml.save_model("autoencoder")



print("Sample results:")
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


with torch.no_grad():
  example_data =example_data.to(device)
  network.to(device)
  output = ml.network(example_data)
    #ENCODOWANIE
    #ml.network.encoder(example_data)

example_data =example_data.cpu()
output_data = output.data.cpu()
print(output_data.shape)
fig = plt.figure()
for i in range(6):
      plt.subplot(2, 6, i + 1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Original: ")
      plt.xticks([])
      plt.yticks([])
      plt.subplot(2, 6, i + 7)
      plt.imshow(output.data[i][0].cpu(), cmap='gray', interpolation='none')
      plt.title("Result:) ")
      plt.xticks([])
      plt.yticks([])
plt.tight_layout()
fig.savefig("../out/autoencoder/comp.png")
plt.close()