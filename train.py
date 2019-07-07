#
# train.py
#

from __future__ import print_function, division

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import argparse
from model import ResNet18
from datasets import MIAS
from PIL import Image

from sampler import ImbalancedDatasetSampler

'''

Arguments.

'''

parser = argparse.ArgumentParser(description='ResNet-MIAS')

parser.add_argument(
	'--load-weights',
	type=str,
	default=None,
	metavar='LW',
	help='load weights from given file'
	)

parser.add_argument(
	'--batch-size',
	type=int,
	default=20,
	metavar='N',
	help='input batch size for training (default: 20)'
	)

parser.add_argument(
	'--epochs',
	type=int,
	default=10,
	metavar='N',
	help='number of epochs to train (default: 10)'
	)

parser.add_argument(
	'--lr',
	type=float,
	default=0.001,
	metavar='LR',
	help='learning rate (default: 0.001)'
	)

parser.add_argument(
	'--cuda',
	action='store_true',
	default=True,
	help='CUDA training'
	)

parser.add_argument(
	'--seed',
	type=int,
	default=1,
	metavar='S',
	help='random seed (default: 1)'
	)

parser.add_argument(
	'--snapshot-folder',
	type=str,
	default='./snapshots',
	metavar='SF',
	help='where to store the snapshots'
	)

parser.add_argument(
	'--data-folder',
	type=str,
	default='',
	metavar='DF',
	help='where to store the datasets'
	)

args = parser.parse_args()

'''

Set a custom seed to ensure reproducibility.

'''

np.random.seed(args.seed)
torch.manual_seed(args.seed)

'''

Specify the running device (CPU or GPU with CUDA support).

'''

device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

'''

Load the dataset.

'''

dataset_folder_path = os.path.join(args.data_folder)

mias_dataset = MIAS(
	transform=transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((128, 128), interpolation=Image.LANCZOS),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.ToTensor()
		])
	)

'''

Prepare the data.

'''

num_samples = len(mias_dataset)
num_classes = mias_dataset.num_classes()
training_batch_size = int(num_samples * 0.7)
validation_batch_size = (num_samples - training_batch_size)

print(f"Training batch size: {str(training_batch_size)}")
print(f"Validation batch size: {str(validation_batch_size)}")

training_set, validation_set = torch.utils.data.random_split(
	mias_dataset,
	[training_batch_size, validation_batch_size]
	)

training_loader = torch.utils.data.DataLoader(
	training_set,
	batch_size=args.batch_size,
	shuffle=False,
	sampler=ImbalancedDatasetSampler(training_set)
	)

validation_loader = torch.utils.data.DataLoader(
	validation_set,
	batch_size=args.batch_size,
	shuffle=True
	)

data_loaders = { 'train': training_loader, 'val': validation_loader }
batch_sizes = { 'train': training_batch_size,'val': validation_batch_size }

'''

Training function.

'''

def train_model(model, criterion, optimizer, scheduler, num_epochs):

	since = time.time()

	best_acc = 0.0
	best_model_wts = copy.deepcopy(model.state_dict())

	for epoch in range(num_epochs):

		epoch_since = time.time()

		for phase in ['train', 'val']:

			running_loss = 0.0
			running_corrects = 0

			if phase == 'train':
				scheduler.step()
				model.train()
			else:
				model.eval()

			for inputs, labels in data_loaders[phase]:

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

			epoch_loss = running_loss / batch_sizes[phase]
			epoch_acc = running_corrects.double() / batch_sizes[phase]
			epoch_time_elapsed = time.time() - epoch_since

			eta = ((num_epochs - epoch) + 1 ) * epoch_time_elapsed

			print('Epoch: {}/{} Phase: {:<5} Loss: {:.4f} Acc: {:.4f} Epoch Time: {:.0f}s ETA: {:.0f}m {:.0f}s'.format(epoch + 1, num_epochs, phase, epoch_loss, epoch_acc, epoch_time_elapsed, eta // 60, eta % 60))

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

	time_elapsed = time.time() - since

	print('')
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	# load best model weights
	model.load_state_dict(best_model_wts)

	return model

'''

Main function.

'''

def main():

	model_ft = ResNet18(num_classes=num_classes)
	model_ft = model_ft.to(device)

	criterion = nn.CrossEntropyLoss()

	optimizer_ft = optim.SGD(
		model_ft.parameters(),
		lr=args.lr,
		momentum=0.9
		)

	dec_lr_scheduler = lr_scheduler.StepLR(
		optimizer_ft,
		step_size=7,
		gamma=0.1
		)

	train_model(
		model_ft,
		criterion,
		optimizer_ft,
		dec_lr_scheduler,
		num_epochs=args.epochs
		)

'''

Call main() on the application default entry point.

'''

if __name__ == "__main__":

	main()
