import torch.nn as nn
import torch
from torchvision.models import ResNet


class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, temp, temp1 = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        test = x * y.expand_as(x)
        return test


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
    )


class CoffeeNet(nn.Module):
    def __init__(self, backend, num_classes=10, device="cuda"):
        super(CoffeeNet, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.backend = backend
        self.layer1 = self.make_layer(512, 256)
        self.skip1 = self.make_skip(512, 256)
        self.layer2 = self.make_layer(256, 128)
        self.skip2 = self.make_skip(256, 128)
        self.layer3 = self.make_layer(128, 64)
        self.skip3 = self.make_skip(128, 64)
        self.fc = nn.Linear(1024, num_classes)
        self.fc.to(device)
        self.dropout = nn.Dropout(0.5)
    
    def make_layer(self, in_channel, out_channel, stride=1, device ="cuda"):
        dance_block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (3,3), stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            SELayer(out_channel, reduction=8),
            nn.Conv2d(out_channel, out_channel, (3,3), stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            SELayer(out_channel, reduction=16),
        )
        dance_block.to(device)
        return dance_block
    def make_skip(self, in_channel, out_channel, device = "cuda"):
        skip = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, (1,1)),
            nn.BatchNorm2d(out_channel)
        )
        skip.to(device)
        return skip

    def encode(self, x):
        return self.backend(x)

    def forward(self, x):
        x = self.encode(x)
        x_skip = self.skip1(x)
        x = self.layer1(x)
        x = self.dropout(x)
        x = x + x_skip
        x_skip = self.skip2(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = x + x_skip
        x_skip = self.skip3(x)
        x = self.layer3(x)
        x = self.dropout(x)
        x = x + x_skip
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class ModifiedModel(nn.Module):
    def __init__(self, backend, inDim, outDim, stride=1, reduction=16, device='cpu', dataset='10', frontend=None):
        super(ModifiedModel, self).__init__()
        self.encoder = backend
        for param in self.encoder.parameters():
            param.requires_grad = False

        if dataset == '10':
            self.classifier = nn.Sequential(
                nn.Conv2d(inDim, outDim, (3, 3), stride=stride, padding=1),
                nn.BatchNorm2d(outDim),
                nn.ReLU(),
                nn.Conv2d(outDim, outDim, (3, 3), stride=1, padding=1),
                nn.BatchNorm2d(outDim),
                SELayer(outDim, reduction),
            )
            self.classifier.to(device)
            self.flatten = nn.Flatten()
            self.relu = nn.ReLU()
            if inDim != outDim:
                self.downsample = nn.Sequential(
                    nn.Conv2d(inDim, outDim, (1, 1), stride=stride, bias=False),
                    nn.BatchNorm2d(outDim)
                )
                self.downsample.to(device)
            else:
                self.downsample = lambda x: x
                self.downsample.to(device)

            for param in self.classifier.parameters():
                nn.init.normal_(param, mean=0, std=0.01)
        else:
            self.classifier = nn.Sequential(
                nn.Conv2d(inDim, outDim, (3, 3), stride=stride, padding=1),
                nn.BatchNorm2d(outDim),
                nn.ReLU(),
                nn.Conv2d(outDim, outDim, (3, 3), stride=1, padding=1),
                nn.BatchNorm2d(outDim),
                SELayer(outDim, reduction),
            )
            self.classifier.to(device)
            self.flatten = nn.Flatten()
            self.relu = nn.ReLU()
            if inDim != outDim:
                self.downsample = nn.Sequential(
                    nn.Conv2d(inDim, outDim, (1, 1), stride=stride, bias=False),
                    nn.BatchNorm2d(outDim)
                )
                self.downsample.to(device)
            else:
                self.downsample = lambda x: x
                self.downsample.to(device)

            for param in self.classifier.parameters():
                nn.init.normal_(param, mean=0, std=0.01)

        self.loss = nn.CrossEntropyLoss()

    def encode(self, X):
        bottleneck = self.encoder(X)
        return bottleneck

    def decode(self, X):
        return self.classifier(X)

    def forward(self, X, train=True):
        encoded_result = self.encode(X)
        residual = self.downsample(encoded_result)         # First SEResNet block
        output1 = self.decode(encoded_result)
        output1 += residual
        output1 = self.relu(output1)
        output1 = self.flatten(output1)

        return output1

        # residual1 = self.downsample(output1)        # Second SEResNet block
        # output2 = self.decode(encoded_result)
        # output2 += residual1
        # output2 = self.relu(output2)
        #
        # residual2 = self.downsample(output2)        # Third SEResNet block
        # output3 = self.decode(encoded_result)
        # output3 += residual2
        # output3 = self.relu(output3)
        #
        # return output3
