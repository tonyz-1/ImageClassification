import torch.nn as nn
import torch
from torchvision.models import ResNet


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)


class modifiedClassifier(nn.Module):
    vgg = nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    ),

    seResNet = nn.Sequential(
        nn.Conv2d(512, 256, (3, 3)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 128, (3, 3)),
        nn.BatchNorm2d(128),
        SELayer(128),
    )


class ModifiedModel(nn.Module):
    def __init__(self, backend, device='cpu', dataset='10'):
        super(ModifiedModel, self).__init__()
        self.encoder = backend
        for param in self.encoder.parameters():
            param.requires_grad = False
        if dataset == '10':
            self.classifier = modifiedClassifier.seResNet
            self.classifier.to(device)
            self.relu = nn.ReLU()
            for param in self.classifier.parameters():
                nn.init.normal_(param, mean=0, std=0.01)
        else:
            self.classifier = modifiedClassifier.seResNet
            self.classifier.to(device)
            self.relu = nn.ReLU()
            for param in self.classifier.parameters():
                nn.init.normal_(param, mean=0, std=0.01)

        self.loss = nn.CrossEntropyLoss()

    def encode(self, X):
        bottleneck = self.encoder(X)
        test = bottleneck.shape
        return bottleneck

    def decode(self, X):
        return self.classifier(X)

    def forward(self, X, train=True):
        residual = X
        encoded_result = self.encode(X)
        output = self.decode(encoded_result)
        if self.downsample is not None:
            residual = self.downsample(X)
        output += residual
        output = self.relu(output)

        return output
