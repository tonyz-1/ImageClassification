from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transform
from torch import optim
import vanillaModel
from prodigyopt import Prodigy
import torch.nn as nn
import torch

import matplotlib.pyplot as plt

device = 'cpu'
encoder_file = './encoder.pth'

# Hyper Parameters
batch_size = 512
weight_decay = 0.1
n_epochs = 10

# Dataset
train_transform_exotic = transform.Compose([transform.RandomCrop(32, padding=4),
                                     transform.RandomHorizontalFlip(),
                                     transform.ToTensor(),
                                     transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
train_transform_vanilla = transform.Compose([transform.ToTensor()])
dataset_train = CIFAR10('./data', download=True, train=True, transform=train_transform_vanilla)
dataset_test = CIFAR10('./data', download=True, train=False, transform=train_transform_vanilla)
train_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

# Model Instantiation
backend = vanillaModel.vggClassifier
backend.vgg.to(device)
backend.vgg.load_state_dict(torch.load(encoder_file, map_location=device))
model = vanillaModel.VanillaModel(backend)
model.train()

#baseline_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
#baseline_model.to(device)
#baseline_model.train()

# Optimizer and Scheduler
optimizer = Prodigy(model.parameters(), lr=1., weight_decay=weight_decay, decouple=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

loss_fn = torch.nn.CrossEntropyLoss()

losses_list = []
for epoch in range(n_epochs):
    loss_epoch = 0.0
    index = 0
    for imgs, labels in train_dl:
        index += 1
        input_ims = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        #output = baseline_model(input_ims)
        #loss = loss_fn(output, labels)

        output = model.forward(input_ims)
        loss = model.loss(output, labels)

        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()
        if index % 5 == 0:
            print(f"Batch {index} loss: {loss}")
    losses_list.append(loss_epoch)
    scheduler.step()
    print(f"Epoch {epoch} loss: {losses_list[epoch]}")
    if epoch % 5 == 0:
        state_dict = model.classifier.state_dict()
        torch.save(state_dict, f"./weights/weight_epoch_{epoch}.pth")


plt.plot(losses_list, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
