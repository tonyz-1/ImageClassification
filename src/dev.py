from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch import optim
import vanillaModel

dataset = CIFAR10('./data', download=True)

# Hyper Parameters
gamma = 1
lr = 1e-4
batch_size = 64

# Dataset
image_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model Instantiation
model = vanillaModel.vggClassifier.vgg()

# Optimizer and Scheduler
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)

