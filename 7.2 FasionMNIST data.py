import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torch.utils
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.datasets import FashionMNIST


mean = np.array([0.5])
std = np.array([0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),  # Resize to 224x224
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),  # Resize to 224x224
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3 channels
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

datasets= { 'train' : FashionMNIST(root= './data', train= 'train', download= True, transform= data_transforms['train']),
          'val' : FashionMNIST(root= './data', train= 'test', download= True, transform= data_transforms['val']) }

# print(len(train_dataset.classes))

dataloaders= { 'train': torch.utils.data.DataLoader(datasets['train'], shuffle= True, batch_size= 6), 
              'val' : torch.utils.data.DataLoader(datasets['val'], shuffle= True, batch_size= 6)}

features, labels= next(iter(dataloaders['train']))
# print(features[0], labels[0])

dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
class_names = datasets['train'].classes
# print(dataset_sizes, class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
# print(inputs, classes)

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title= [class_names[x] for x in classes])

# train model
def train_model(model, criterion, optimizer, scheduler, num_epochs= 25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, len(class_names))

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr= 0.001)

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma= 0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=25)

print("*"*150)


