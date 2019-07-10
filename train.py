#
# train.py
#

from __future__ import print_function, division

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import errno
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
from model import ResNet18, ResNet34, ResNet50, VGG
from datasets import MIAS
from PIL import Image

import seaborn as sns
import pandas as pd
import sklearn.metrics as sm
from sampler import ImbalancedDatasetSampler
from sklearn.utils.multiclass import unique_labels

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
	'--network',
	type=str,
	default='ResNet18',
	metavar='N',
	help='network (default: ResNet18)'
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
	'--load',
	type=int,
	default=2,
	metavar='LOAD',
	help='MIAS load (default: 2)'
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

print('')

dataset_folder_path = os.path.join(args.data_folder)

try:
	os.makedirs(dataset_folder_path)
except OSError as e:
	if e.errno != errno.EEXIST:
		raise

mias_dataset = MIAS(
	load = args.load,
	transform=transforms.Compose([
		transforms.ToPILImage(),
		transforms.Resize((128, 128), interpolation=Image.LANCZOS),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.ToTensor()
		])
	)

mias_dataset.plot_class_distribution(dataset_folder_path)

'''

Prepare the data.

'''

num_samples = len(mias_dataset)
all_classes = mias_dataset.all_classes()
num_classes = mias_dataset.num_classes()

training_batch_size = int(num_samples * 0.8)
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

print('')
print('batch size:')
print(args.batch_size)
print('')

data_loaders = { 'train': training_loader, 'val': validation_loader }
batch_sizes = { 'train': training_batch_size,'val': validation_batch_size }

'''

xxx

'''

def generate_visualizations(
	t_accuracies,
	t_losses,
	v_accuracies,
	v_losses,
	confusion_matrice,
	labels,
	folder=None
	):

	cm_sum = np.sum(confusion_matrice, axis=1, keepdims=True)
	cm_percentuals = confusion_matrice / cm_sum * 100
	annot = np.empty_like(confusion_matrice).astype(str)
	nrows, ncols = confusion_matrice.shape

	for i in range(nrows):
		for j in range(ncols):
			c = confusion_matrice[i, j]
			p = cm_percentuals[i, j]
			if c == 0:
				annot[i, j] = ''
			else:
				annot[i, j] = '%.1f%%' % p

	cm = pd.DataFrame(confusion_matrice, index=labels, columns=labels)
	cm.index.name = 'Actual'
	cm.columns.name = 'Predicted'

	plt.figure()
	plt.margins(0)
	plt.xlabel('epoch')
	plt.ylabel('accuracy')
	plt.plot(t_accuracies, 'b', label='training')
	plt.plot(v_accuracies, 'g', label='validation')
	plt.grid(True)
	plt.legend(loc="lower right")
	if folder == None:
		plt.show()
	else:
		plt.savefig(folder + '/accuracies-tt.png', bbox_inches = 'tight')

	plt.figure()
	plt.margins(0)
	plt.xlabel('epoch')
	plt.ylabel('loss')
	plt.plot(t_losses, 'b', label='training')
	plt.plot(v_losses, 'g', label='validation')
	plt.grid(True)
	plt.legend(loc="upper right")
	if folder == None:
		plt.show()
	else:
		plt.savefig(folder + '/losses-tt.png', bbox_inches = 'tight')

	plt.figure()
	sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
	plt.xlabel('predicted')
	plt.ylabel('actual')
	if folder == None:
		plt.show()
	else:
		plt.savefig(folder + '/confusion-matrix.png')

'''

Training function.

'''

def train_model(model, criterion, optimizer, scheduler, num_epochs):

	def train(data):

		running_loss = 0.0
		running_corrects = 0

		scheduler.step()
		model.train()

		for inputs, labels in data:

			inputs = inputs.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()

			with torch.set_grad_enabled(True):

				outputs = model(inputs)
				_, preds = torch.max(outputs, 1)
				loss = criterion(outputs, labels)

				loss.backward()
				optimizer.step()

			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)

		batch_size = batch_sizes['train']
		epoch_loss = running_loss / batch_size
		epoch_acc = running_corrects.double() / batch_size

		return epoch_acc, epoch_loss

	def test(data):

		running_loss = 0.0
		running_corrects = 0

		running_targets = []
		running_predictions = []

		model.eval()

		for inputs, labels in data:

			inputs = inputs.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()

			with torch.set_grad_enabled(False):

				outputs = model(inputs)
				_, preds = torch.max(outputs, 1)
				loss = criterion(outputs, labels)

				running_targets += list(labels.cpu().numpy())
				running_predictions += list(preds.cpu().numpy())

			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)

		batch_size = batch_sizes['train']
		epoch_loss = running_loss / batch_size
		epoch_acc = running_corrects.double() / batch_size

		return epoch_acc, epoch_loss, (running_targets, running_predictions)

	# ...

	since = time.time()

	t_accuracies, t_losses = [], []
	v_accuracies, v_losses = [], []
	best_acc = -1
	best_results = ()
	best_model_wts = copy.deepcopy(model.state_dict())

	for epoch in range(num_epochs):

		epoch_since = time.time()

		t_init_time = time.time()
		t_acc, t_loss = train(data_loaders['train'])
		t_duration = time.time() - t_init_time
		t_eta = ((num_epochs - epoch) + 1 ) * t_duration

		print('Epoch: {}/{} Phase: {:<5} Loss: {:.4f} Acc: {:.4f} Epoch Time: {:.0f}s ETA: {:.0f}m {:.0f}s'.format(
			epoch + 1,
			num_epochs,
			'training',
			t_loss,
			t_acc,
			t_duration,
			t_eta // 60,
			t_eta % 60
			))

		t_accuracies.append(t_acc)
		t_losses.append(t_loss)

		v_init_time = time.time()
		v_acc, v_loss, v_results = test(data_loaders['val'])
		v_duration = time.time() - v_init_time
		v_eta = ((num_epochs - epoch) + 1 ) * t_duration

		print('Epoch: {}/{} Phase: {:<5} Loss: {:.4f} Acc: {:.4f} Epoch Time: {:.0f}s ETA: {:.0f}m {:.0f}s'.format(
			epoch + 1,
			num_epochs,
			'validation',
			v_loss,
			v_acc,
			v_duration,
			v_eta // 60,
			v_eta % 60
			))

		if v_acc > best_acc:

			best_acc = v_acc
			best_results = v_results
			best_model_wts = copy.deepcopy(model.state_dict())

		v_accuracies.append(v_acc)
		v_losses.append(v_loss)

	time_elapsed = time.time() - since
	labels = [all_classes[i] for i in unique_labels(best_results[0], best_results[1])]
	confusion_matrix = sm.confusion_matrix(best_results[0], best_results[1])

	model.load_state_dict(best_model_wts)

	generate_visualizations(
		t_accuracies,
		t_losses,
		v_accuracies,
		v_losses,
		confusion_matrix,
		labels,
		folder=dataset_folder_path
		)

	print('')
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best val Acc: {:4f}'.format(best_acc))

	return model

'''

Main function.

'''

def main():

	print('')

	if args.network == 'ResNet18':

		model_ft = ResNet18(num_classes=num_classes)
		model_ft = model_ft.to(device)

	elif args.network == 'ResNet34':

		model_ft = ResNet34(num_classes=num_classes)
		model_ft = model_ft.to(device)

	elif args.network == 'ResNet50':

		model_ft = ResNet50(num_classes=num_classes)
		model_ft = model_ft.to(device)

	elif args.network == 'VGG11':

		model_ft = VGG('VGG11', 1, 7)
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

	print('')


'''

Call main() on the application default entry point.

'''

if __name__ == "__main__":

	main()
