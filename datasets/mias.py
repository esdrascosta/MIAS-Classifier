#
# mias.py
#

import os
import re
import json
import gzip
import errno
import tarfile

import random
import shutil
from PIL import Image

from six.moves import urllib

import numpy as np
import torch.utils.data as data

'''

Class grouping information about an instance of the MIAS dataset.

'''

class MIAS(data.Dataset):

	root_folder_name = 'data'
	raw_data_folder_name = 'all-mias'

	url_dataset = 'http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz'

	'''

	Class initialization.

	'''

	def __init__(self, transform=None, target_transform=None, load=2):

		self.transform = transform
		self.target_transform = target_transform

		# Download the dataset if specified and needed.

		self.__download__(self.url_dataset)

		# Parse the dataset information.

		p, s, l = self.__parse_data__(self.raw_data_folder_name)

		if load == 0:

			# Use raw data, full sized images and no data augmentation.

			self.patients = p
			self.samples = s
			self.labels_enumeration = l

		elif load == 1:

			# Use full sized images and data augmentation.

			p, s, l = self.__simple_data_augmentation__(p, l)

			self.patients = p
			self.samples = s
			self.labels_enumeration = l

		else:

			# Use images containing the ROIs and data augmentation.

			p, s, l = self.__simple_data_augmentation__(p, l)
			p, s, l = self.__extend_data_augmentation__(p, l)

			self.patients = p
			self.samples = s
			self.labels_enumeration = l

	'''

	Returns the number of samples our dataset has.

	'''

	def __len__(self):

		return len(self.samples)

	'''

	Returns a tuple (image, target) when the getitem accessor is used.

	'''

	def __getitem__(self, index):

		path, target = self.samples[index]
		sample = self._read_pgm(path)

		if self.transform is not None:

			sample = self.transform(sample)

		if self.target_transform is not None:

			target = self.target_transform(target)

		return sample, target

	'''

	xxx

	'''

	def _read_pgm(self, filename, byteorder='>'):

		"""
		Return image data from a raw PGM file as a numpy array.
		Format specification: http://netpbm.sourceforge.net/doc/pgm.html
		"""

		with open("{}.pgm".format(filename), 'rb') as f:

			buffer = f.read()

		try:

			header, width, height, maxval = re.search(
				b"(^P5\s(?:\s*#.*[\r\n])*"
				b"(\d+)\s(?:\s*#.*[\r\n])*"
				b"(\d+)\s(?:\s*#.*[\r\n])*"
				b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()

		except AttributeError:

			raise ValueError("Not a raw PGM file: '%s'" % filename)

		return np.frombuffer(buffer,
			dtype='u1' if int(maxval) < 256 else byteorder+'u2',
			count=int(width)*int(height),
			offset=len(header)
			).reshape(( int(height), int(width) ))

	'''

	Downloads the database from a remote location.

	'''

	def __download__(self, url):

		print('Downloading: ' + url)

		download_destination_path = self.root_folder_name
		download_file_name = url.rpartition('/')[2]
		download_file_path = os.path.join(download_destination_path, download_file_name)

		download_folder_name = download_file_name.replace('.tar.gz', '')
		download_folder_path = os.path.join(download_destination_path, download_folder_name)

		try:
			os.makedirs(download_destination_path)
		except OSError as e:
			if e.errno == errno.EEXIST:
				print('Data folder exists, skipping download.')
			return

		data = urllib.request.urlopen(url)

		with open(download_file_path, 'wb') as f:

			f.write(data.read())

		tf = tarfile.open(download_file_path)
		tf.extractall(download_folder_path)
		tf.close()
		os.unlink(download_file_path)

	'''

	Extracts information from the dataset 'Infos.txt' file and parses it into
	an array of dictionaries. Each array item has the 'left' and 'right' keys,
	referencing the images for the left and right mammograms. Each dictionary
	contains the image filename,  background tissue, class of abnormality,
	severity of abnormality, the coordinates and radius of the abnormality.

	'''

	def __parse_mias_infos__(self, folder_path_dataset):

		regex = r"(mdb\d{3})\W([FGD])\W(CALC|CIRC|SPIC|MISC|ARCH|ASYM|NORM)\W?([BM])?\W?((\d{1,3})\W(\d{1,3})\W(\d{1,3})|(\*NOTE\W3\*))?"

		patients = []
		patient_index = 0
		patient_rleft = False

		file_names = []
		file_labels = []

		infos_filename = 'Info.txt'
		infos_filepath = os.path.join(folder_path_dataset, infos_filename)
		infos_file = open(infos_filepath)

		for line in infos_file:

			matches = re.finditer(regex, line)

			for matchNum, match in enumerate(matches, start=1):

				img_filename = match.group(1)
				bck_tissue = match.group(2)
				abn_class = match.group(3)

				if abn_class != 'NORM':

					abn_severity = match.group(4)
					line_litems = match.group(5)
					abn_radius = 0
					abn_coordinates = (0, 0)
					abn_has_coordinates = False

					if line_litems != None and line_litems != '*NOTE 3*':

						items = line_litems.split(' ')
						abn_has_coordinates = True
						abn_coordinates = (int(items[0]), int(items[1]))
						abn_radius = int(items[2])

				if patient_rleft == False:

					patients.append({'left':{}, 'right':{}})

					cside = 'left'
					cindex = patient_index
					patient_rleft = True

				else:

					cside = 'right'
					cindex = patient_index
					patient_rleft = False
					patient_index = patient_index + 1

				file_names.append(img_filename)
				file_labels.append(abn_class)

				patients[cindex][cside]['img_filename'] = img_filename
				patients[cindex][cside]['bck_tissue'] = bck_tissue
				patients[cindex][cside]['abn_class'] = abn_class
				patients[cindex][cside]['abn_severity'] = abn_severity
				patients[cindex][cside]['abn_radius'] = abn_radius
				patients[cindex][cside]['abn_coordinates'] = abn_coordinates
				patients[cindex][cside]['abn_has_coordinates'] = abn_has_coordinates

		return file_names, file_labels, patients

	'''

	Parses the information available in the dataset `Infos.txt` file.

	'''

	def __parse_data__(self, folder_name_dataset):

		folder_path_dataset = os.path.join(self.root_folder_name, folder_name_dataset)
		file_names, file_labels, patients = self.__parse_mias_infos__(folder_path_dataset)

		file_paths = [ os.path.join(folder_path_dataset, fname) for fname in file_names ]
		labels_enumeration = { v:i for i, v in enumerate( list({ i for i in file_labels }) ) }
		targets = [ labels_enumeration[label] for label in file_labels ]

		patients = patients
		samples = list(zip(file_paths, targets))
		labels_enumeration = labels_enumeration

		return patients, samples, labels_enumeration

	'''

	Performs a simple dataset augmentation. Each image is rotated three times by
	steps of 90 degrees, flipped horizontally and vertically (five new copies).

	'''

	def __simple_data_augmentation__(self, patients, labels_enumeration):

		p_index = 0
		n_samples = []
		n_patients = []

		folder_name_dataset = 'all-augmented'
		folder_path_dataset = os.path.join(self.root_folder_name, folder_name_dataset)

		infos_filename = 'infos.json'
		infos_filepath = os.path.join(folder_path_dataset, infos_filename)

		if os.path.isdir(folder_path_dataset) == True:

			infos_file = open(infos_filepath, "r")
			patients = json.load(infos_file)

			for patient in patients:

				for side in ['left', 'right']:

					abn_class = patient[side]['abn_class']
					abn_class_enumeration = labels_enumeration[abn_class]
					img_filename = patient[side]['img_filename']
					img_path = os.path.join(folder_path_dataset, img_filename)

					n_samples.append((img_path, abn_class_enumeration))

			return patients, n_samples, labels_enumeration

		try:
			os.makedirs(folder_path_dataset)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

		infos_file = open(infos_filepath, "w+")

		for patient in patients:

			n_patients.append({})
			n_patients.append({})
			n_patients.append({})
			n_patients.append({})
			n_patients.append({})

			for side in ['left', 'right']:

				o_abn_coordinates = patient[side]['abn_coordinates']
				o_coordinate_x = o_abn_coordinates[0]
				o_coordinate_y = o_abn_coordinates[1]
				o_abn_class = patient[side]['abn_class']
				o_ab_class_enum = labels_enumeration[o_abn_class]
				o_img_filename = patient[side]['img_filename']
				o_img_path = os.path.join(self.root_folder_name, self.raw_data_folder_name, o_img_filename)+'.pgm'

				shutil.copy(o_img_path, os.path.join(folder_path_dataset, o_img_filename)+'.pgm')

				n_samples_img_path = os.path.join(folder_path_dataset, o_img_filename)
				n_samples.append((n_samples_img_path, o_ab_class_enum))

				for i in range(5):

					index = p_index + i

					n_img_filename = o_img_filename + '-' + str(index) + '-' + side
					n_img_path = os.path.join(folder_path_dataset, n_img_filename)+'.pgm'

					if i == 0:
						n_coordinate_x = (1024 - o_coordinate_x)
						n_coordinate_y = o_coordinate_y
						Image.open(o_img_path).transpose(Image.FLIP_LEFT_RIGHT).save(n_img_path)
					elif i == 1:
						n_coordinate_x = o_coordinate_x
						n_coordinate_y = (1024 - o_coordinate_y)
						Image.open(o_img_path).transpose(Image.FLIP_TOP_BOTTOM).save(n_img_path)
					elif i == 2:
						n_coordinate_x = (1024 - o_coordinate_y)
						n_coordinate_y = o_coordinate_x
						Image.open(o_img_path).rotate(90).save(n_img_path)
					elif i == 3:
						n_coordinate_x = (1024 - o_coordinate_x)
						n_coordinate_y = (1024 - o_coordinate_y)
						Image.open(o_img_path).rotate(180).save(n_img_path)
					elif i == 4:
						n_coordinate_x = o_coordinate_y
						n_coordinate_y = (1024 - o_coordinate_x)
						Image.open(o_img_path).rotate(270).save(n_img_path)

					n_patients[index][side] = {}
					n_patients[index][side]['img_filename'] = n_img_filename
					n_patients[index][side]['abn_coordinates'] = (n_coordinate_x, n_coordinate_y)
					n_patients[index][side]['bck_tissue'] = patient[side]['bck_tissue']
					n_patients[index][side]['abn_class'] = patient[side]['abn_class']
					n_patients[index][side]['abn_severity'] = patient[side]['abn_severity']
					n_patients[index][side]['abn_radius'] = patient[side]['abn_radius']
					n_patients[index][side]['abn_has_coordinates'] = patient[side]['abn_has_coordinates']

					n_samples_img_path = os.path.join(folder_path_dataset, n_img_filename)
					n_samples.append((n_samples_img_path, o_ab_class_enum))

			p_index = p_index + 5

		json.dump((patients + n_patients), infos_file, indent=4)

		return (patients + n_patients), n_samples, labels_enumeration

	'''

	xxx

	'''

	def __extend_data_augmentation__(self, patients, labels_enumeration):

		p_index = 0
		n_samples = []
		n_patients = []

		folder_name_dataset = 'all-augmented-cropped'
		folder_path_dataset = os.path.join(self.root_folder_name, folder_name_dataset)

		infos_filename = 'infos.json'
		infos_filepath = os.path.join(folder_path_dataset, infos_filename)

		if os.path.isdir(folder_path_dataset) == True:

			infos_file = open(infos_filepath, "r")
			patients = json.load(infos_file)

			for patient in patients:

				abn_class = patient['abn_class']
				abn_class_enumeration = labels_enumeration[abn_class]
				img_filename = patient['img_filename']
				img_path = os.path.join(folder_path_dataset, img_filename)

				n_samples.append((img_path, abn_class_enumeration))

			return patients, n_samples, labels_enumeration

		try:
			os.makedirs(folder_path_dataset)
		except OSError as e:
			if e.errno != errno.EEXIST:
				raise

		infos_file = open(infos_filepath, "w+")

		for patient in patients:

			for side in ['left', 'right']:

				abn_class = patient[side]['abn_class']
				abn_class_enum = labels_enumeration[abn_class]

				abn_has_coordinates = patient[side]['abn_has_coordinates']
				abn_coordinates = patient[side]['abn_coordinates']
				coordinate_x = abn_coordinates[0]
				coordinate_y = abn_coordinates[1]
				abn_radius = patient[side]['abn_radius']

				o_img_filename = patient[side]['img_filename']
				o_img_path = os.path.join(self.root_folder_name, 'all-augmented', o_img_filename)+'.pgm'

				o_img_filename = patient[side]['img_filename']
				o_img_path = os.path.join(self.root_folder_name, 'all-augmented', o_img_filename)+'.pgm'
				n_img_path = os.path.join(folder_path_dataset, o_img_filename)+'.pgm'

				if abn_class == 'NORM':

					num_windows = 1
					window_size = 128

					for i in range(num_windows):

						n_img_filename = (o_img_filename + '-' + str(i))
						n_img_path = os.path.join(folder_path_dataset, n_img_filename)+'.pgm'

						x0 = max(0, 256 + random.randint(1, 384))
						y0 = max(0, 256 + random.randint(1, 384))
						xN = min((x0+window_size), 1024)
						yN = min((y0+window_size), 1024)
						window_coordinates = (x0, y0, xN, yN)

						Image.open(o_img_path).crop(window_coordinates).resize((128, 128)).save(n_img_path)

						n_patients.append({})
						n_patients[p_index] = {}
						n_patients[p_index]['img_filename'] = n_img_filename
						n_patients[p_index]['abn_class'] = abn_class
						n_samples.append((n_img_path, abn_class_enum))
						p_index = p_index + 1

				elif abn_has_coordinates == True:

					window_size = max(128, (abn_radius*2)) * 1.2
					x0 = max(0, coordinate_x - (window_size / 2))
					y0 = max(0, (1024-coordinate_y) - (window_size / 2))
					xN = min((x0+window_size), 1024)
					yN = min((y0+window_size), 1024)
					window_coordinates = (x0, y0, xN, yN)

					Image.open(o_img_path).crop(window_coordinates).resize((128, 128)).save(n_img_path)

					n_patients.append({})
					n_patients[p_index] = {}
					n_patients[p_index]['img_filename'] = o_img_filename
					n_patients[p_index]['abn_class'] = abn_class
					n_samples.append((n_img_path, abn_class_enum))
					p_index = p_index + 1

		json.dump(n_patients, infos_file, indent=4)

		return n_patients, n_samples, labels_enumeration

	'''

	xxx

	'''

	def plot_class_distribution(self):

		print('Plotting class distribution.')

		labels = list(self.labels_enumeration.keys())
		samples = [sample[1] for sample in self.samples]
		counts = { labels[i]:samples.count(i) for i in range(0, len(labels))}

		items = [(label, counts[label]) for label in labels]
		items = sorted(items, key=lambda x: x[1])

		x_labels = [item[0] for item in items]
		y_pos = np.arange(len(x_labels))
		y_values = [item[1] for item in items]

		plt.bar(y_pos, y_values, align='center', alpha=0.8)
		plt.xticks(y_pos, x_labels)
		plt.ylabel('Ocorrências')
		plt.title('Distribuição do Banco por Classe')
		plt.show()

	'''

	Returns the number of classes on the parsed data.

	'''

	def num_classes(self):

		return len(self.labels_enumeration)

	'''

	Translates a given integer to the original class name.

	'''

	def get_class(self, id):

		for name, i in self.labels_enumeration.items():

			if i == id:

				return name

'''

xxx

'''

if __name__ == "__main__":

	# Hack to make matplotlib work on macOS under a virtual enviorment.

	import matplotlib
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt

	# Initialize the dataset.
	# If the data doesnt't exist, the class will download it.
	# If augment is True and data aumentation wasn't performed yet, it will.

	dataset = MIAS()

	# Plot the samples distribution per class.

	dataset.plot_class_distribution()
