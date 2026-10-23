import torch
import time
import torchvision
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

from classifier.mnist_cnn import MnistCNN
from common.ml_wrapper import MLWrapper

print(torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name())
data_loader_kwargs ={'pin_memory': True}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
base_path = ".."
datasets_path = "../datasets/"
n_epochs = 30
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

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


network = MnistCNN()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
ml = MLWrapper(network, optimizer, base_path, "out/mnist", device)
ml.summary((1,28,28))
model_file_name = "conv_network_0.2_model.pth"

if not Path(f'{base_path}/out/mnist/{model_file_name}').exists():
    print("Start training")
    globalT= time.time()
    for epoch in range(1, n_epochs + 1):
      ml.train(epoch, train_loader, test_loader)
    globalT = time.time() - globalT
    print("Total time of training(s):{:.4f}".format(globalT))
    ml.save_model("conv_network")
else:
    print("Loading model from file:", model_file_name)
    ml = MLWrapper.load_model(base_path, "out/mnist", model_file_name, network, optimizer=optimizer, device=device)

print("Test on training set:")
ml.test(train_loader)
print("Test on test set:")
ml.test(test_loader)

ml.training_plot()

print("Sample results:")
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


with torch.no_grad():
  example_data =example_data.to(device)
  network.to(device)
  output = ml.network(example_data)

example_data =example_data.cpu()
fig = plt.figure()
for i in range(6):
      plt.subplot(2, 3, i + 1)
      plt.tight_layout()
      plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
      plt.title("Prediction: {}".format(
          output.data.max(1, keepdim=True)[1][i].item()))
      plt.xticks([])
      plt.yticks([])
#fig.show()
fig.savefig(f"{base_path}/out/mnist/train.png")
plt.close()