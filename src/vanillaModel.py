import torch.nn as nn
import torch

class vggClassifier(nn.Module):
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
    classifier = nn.Sequential(
        nn.Conv2d(512, 256, (1, 1)),
        nn.Conv2d(256, 128, (1, 1)),
        nn.Flatten(),
        nn.Linear(2048, 10)
    )
    classifier_100 = nn.Sequential(
        nn.Conv2d(512, 256, (1, 1)),
        nn.Conv2d(256, 128, (1, 1)),
        nn.Flatten(),
        nn.Linear(2048, 100)
    )


class VanillaModel(nn.Module):
    def __init__(self, backend, frontend=None, device='cpu', dataset='10'):
        super(VanillaModel, self).__init__()
        self.encoder = backend
        for param in self.encoder.parameters():
            param.requires_grad = False
        if dataset == '10':
            if frontend == None:
                self.classifier = vggClassifier.classifier
                self.classifier.to(device)
                for param in self.classifier.parameters():
                    nn.init.normal_(param, mean=0, std=0.01)
            else:
                self.classifier = frontend
        else:
            if frontend == None:
                self.classifier = vggClassifier.classifier_100
                self.classifier.to(device)
                for param in self.classifier.parameters():
                    nn.init.normal_(param, mean=0, std=0.01)
            else:
                self.classifier = frontend

        self.loss = nn.CrossEntropyLoss()

    def encode(self, X):
        bottleneck = self.encoder(X)
        test = bottleneck.shape
        return bottleneck

    def decode(self, X):
        return self.classifier(X)
    
    def forward(self, X, train=True):
        encoded_result = self.encode(X)
        output_tensor = self.decode(encoded_result)
        #sm = nn.Softmax(dim=1)
        #classified = sm(output_tensor)
        return output_tensor

