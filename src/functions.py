import torch

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
            temp, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += loss_fn(outputs, labels).cpu().item() / len(dl)
    return 100* correct/total, loss