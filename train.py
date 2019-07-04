from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
from model import ResNet18
from datasets import MIAS
from PIL import Image

parser = argparse.ArgumentParser(description='ResNet-MIAS')
parser.add_argument('--load-weights', type=str, default=None, metavar='LW',
                    help='load weights from given file')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 20)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--snapshot-folder', type=str, default='./snapshots', metavar='SF',
                    help='where to store the snapshots')
parser.add_argument('--data-folder', type=str, default='./data', metavar='DF',
                    help='where to store the datasets')
args = parser.parse_args()

# REPRODUCIBILITY
torch.manual_seed(args.seed)
np.random.seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

# load dataset
data_path = os.path.join(args.data_folder)
full_dataset = MIAS(data_path, download=False, 
            transform=transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128), interpolation=Image.LANCZOS), 
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ]))    
num_classes = len(full_dataset.labels_info)
train_size = int(len(full_dataset) * .7) 
val_size = len(full_dataset) - train_size

print(f"Train Size: {str(train_size)}")
print(f"Val Size: {str(val_size)}")

train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, 
                batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, 
                batch_size=args.batch_size, shuffle=True)

dataloaders = {
  'train': train_loader,
  'val': val_loader
}

dataset_sizes = {
  'train': train_size,
  'val': val_size
}

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
      # Each epoch has a training and validation phase
      epoch_since = time.time()
      for phase in ['train', 'val']:
           
        if phase == 'train':
          scheduler.step()
          model.train()  # Set model to training mode
        else:
          model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
          inputs = inputs.to(device)
          labels = labels.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()

          # forward
          # track history if only in train
          with torch.set_grad_enabled(phase == 'train'):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

              # backward + optimize only if in training phase
            if phase == 'train':
              loss.backward()
              optimizer.step()

          # statistics
          running_loss += loss.item() * inputs.size(0)
          running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        epoch_time_elapsed = time.time() - epoch_since
        eta = ((num_epochs - epoch) + 1 ) * epoch_time_elapsed
        print('Epoch: {}/{} Phase: {:<5} Loss: {:.4f} Acc: {:.4f} Epoch Time: {:.0f}s ETA: {:.0f}m {:.0f}s'.format(
          epoch + 1, num_epochs, phase, epoch_loss, epoch_acc, epoch_time_elapsed, eta // 60, eta % 60))

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

def main(): 
  model_ft = ResNet18(num_classes=num_classes)
  print(model_ft)
  model_ft = model_ft.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer_ft = optim.SGD(model_ft.parameters(), lr=args.lr, momentum=0.9)

  # Decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

  train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=args.epochs)

if __name__ == "__main__":
    main()
    