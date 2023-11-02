from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transform
from torch import optim
import vanillaModel
from prodigyopt import Prodigy
import torch
import argparse
import functions
from train import train


parser = argparse.ArgumentParser()
parser.add_argument('-e', type=int, help='epochs')
parser.add_argument('-b', type=int, help='batch size')
parser.add_argument('-l', type=str, help='encoder weight file')
parser.add_argument('-s', type=str, help='decoder weight file')
parser.add_argument('-p', type=str, help='Loss output image')
parser.add_argument('-cuda', type=str, help='[Y/N] for gpu usage')
parser.add_argument('-vanilla', type=str, help='Vanilla model [Y/N]')
opt = parser.parse_args()

encoder_file = opt.l
decoder_file = opt.s
loss_file = opt.p

device = 'cpu'

if (opt.cuda == 'y' or opt.cuda == 'Y') and (torch.cuda.is_available()):
    use_cuda = True
    device = 'cuda'
    torch.cuda.empty_cache()
    m = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)

encoder_file = './encoder.pth'

# Model Instantiation
backend = vanillaModel.vggClassifier.vgg
backend.to(device)
backend.load_state_dict(torch.load(encoder_file, map_location=device))

if decoder_file != None:
    frontend = vanillaModel.vggClassifier.classifier
    frontend.to(device)
    frontend.load_state_dict(torch.load(decoder_file, map_location=device))
    model = vanillaModel.VanillaModel(backend, frontend)
else:
    model = vanillaModel.VanillaModel(backend)

#model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
#model.to(device)
#model.train()

# Hyper Parameters
weight_decay = 0.1
n_epochs = opt.e
batch_size = opt.b

# Optimizer and Scheduler
if opt.vanilla == 'y' or opt.vanilla == 'Y':
    optimizer = optim.Adam(model.parameters(), 1, weight_decay=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 1)
else:
    optimizer = Prodigy(model.parameters(), lr=1., weight_decay=weight_decay, decouple=True, safeguard_warmup=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

# Loss Function
loss_fn = torch.nn.CrossEntropyLoss()

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

train(model, n_epochs, train_dl, test_dl, device, optimizer, scheduler, loss_fn, loss_file)
