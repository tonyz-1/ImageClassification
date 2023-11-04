import torch
import vanillaModel
import modifiedModel
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transform

# Implementation from https://colab.research.google.com/drive/1TrhEfI3stJ-yNp7_ZxUAtfWjj-Qe_Hym?usp=sharing#scrollTo=YJLiIn8canrB
# A prodigy implementation
def accuracy_loss(model, dl, device, loss_fn):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in dl:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            vals, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += loss_fn(outputs, labels).cpu().item() / len(dl)
    return 100*correct/total, loss

def hit(label, predicted):
    count = 0
    for index in range(label.size(0)):
        test = predicted[index]
        if label[index] in predicted[index]:
            count += 1
    return count

# Returns top k error rate as a percentage
def top_k_error(k, model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    batch_num = 0
    with torch.no_grad():
        for data in dataloader:
            batch_num += 1
            print(f"Batch:{batch_num}")
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            vals, predicted = torch.topk(outputs.data, k, dim=1)
            if k > 1:
                correct += hit(labels, predicted)
            else:
                predicted = torch.flatten(predicted)                
                correct += (labels == predicted).sum().item()
            total += labels.size(0)
    return 100*(total-correct)/total

# Returns instance of model from vanillaModel
def init_vanilla_model(encoder_file, device, decoder_file=None):
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
    return model

# Returns instance of modified model
def init_modified_model(encoder_file, device, frontend_weights=None):
    backend = modifiedModel.modifiedClassifier.vgg
    backend.to(device)
    backend.load_state_dict(torch.load(encoder_file, map_location=device))
    model = modifiedModel.ModifiedModel(backend)
    return model

# Returns train and test dataloaders
def init_dataloaders(batch_size, augmented=False, padding=4, crop=32, dataset="100"):
    if augmented:
        train_transform = transform.Compose([transform.RandomCrop(crop, padding=padding),
                                        transform.RandomHorizontalFlip(),
                                        transform.ToTensor(),
                                        transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    else:
        train_transform = transform.Compose([transform.ToTensor()])
    if dataset == '10':
        dataset_train = CIFAR10('./data', download=True, train=True, transform=train_transform)
        dataset_test = CIFAR10('./data', download=True, train=False, transform=train_transform)
        train_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        return train_dl, test_dl
    else:
        dataset_train = CIFAR100('./data', download=True, train=True, transform=train_transform)
        dataset_test = CIFAR100('./data', download=True, train=False, transform=train_transform)
        train_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        return train_dl, test_dl