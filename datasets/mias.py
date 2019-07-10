import torch.utils.data as data
from six.moves import urllib
from PIL import Image
import numpy as np
import shutil
import tarfile
import errno
import gzip
import re
import os
import csv


class MIAS(data.Dataset):

    raw_folder = 'raw'
    processed_folder = 'processed'
    dataset_file = 'dataset.pt'
    dataset_url = 'http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz'

    def __init__(self, root, transform=None, target_transform=None, download=True, augment=True):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        # TODO proces files to build an unique dataset file
        self.parse_data(augment)

    """
    Returns the number of samples
    """

    def __len__(self):
        return len(self.samples)

    """
    Returns a tuple (image, target) when the getitem accessor is used.
    """

    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self._read_pgm(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def parse_data(self, augument=True):

        url = self.dataset_url
        filename = url.rpartition('/')[2]
        folder_path = os.path.join(
            self.root,
            self.raw_folder,
            filename.replace('.tar.gz', '')
        )
        file_names, labels = self._parse_infos(folder_path)

        if augument:
            orign_folder_path = folder_path
            folder_path = os.path.join(
                self.root,
                self.raw_folder,
                'all-mias-augmented'
            )
            file_names, labels = self._simple_data_augumentation(
                file_names, labels, orign_folder_path, folder_path)

        self.raw_labels = labels
        self.file_paths = [os.path.join(folder_path, file_name)
                           for file_name in file_names]
        self.labels_info = {v: i for i,
                            v in enumerate(list({i for i in labels}))}
        self.targets = [self.labels_info[label] for label in labels]
        self.samples = list(zip(self.file_paths, self.targets))

    def _simple_data_augumentation(self, file_names, labels, orign_folder_path, destination_folder_path):

        try:
            os.makedirs(destination_folder_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        n_names = []
        n_labels = []

        if os.path.exists(os.path.join(destination_folder_path, 'augment_info.csv')):
            with open(os.path.join(destination_folder_path, 'augment_info.csv')) as f:
                meta_info = csv.reader(f)
                for row in meta_info:
                    n_names.append(row[0])
                    n_labels.append(row[1])

            return n_names, n_labels

        for i, orign_img_file in enumerate(file_names):

            shutil.copy(
                os.path.join(orign_folder_path, orign_img_file+'.pgm'),
                os.path.join(destination_folder_path, orign_img_file+'.pgm')
            )
            n_names.append(orign_img_file)
            n_labels.append(labels[i])

            curr_img_path = os.path.join(
                destination_folder_path, orign_img_file+'.pgm')

            if labels[i] != 'NORM':

                # flip image from left to right
                n_img_flr_name = orign_img_file+'_flr'
                n_img_flr_path = os.path.join(
                    destination_folder_path, n_img_flr_name + '.pgm')
                Image.open(curr_img_path).transpose(
                    Image.FLIP_LEFT_RIGHT).save(n_img_flr_path)

                n_names.append(n_img_flr_name)
                n_labels.append(labels[i])

                # flip image from top to bottom
                n_img_ftb_name = orign_img_file+'_ftb'
                n_img_ftb_path = os.path.join(
                    destination_folder_path,
                    n_img_ftb_name + '.pgm'
                )
                Image.open(curr_img_path).transpose(
                    Image.FLIP_TOP_BOTTOM).save(n_img_ftb_path)

                n_names.append(n_img_ftb_name)
                n_labels.append(labels[i])

                # rotate 45 degress
                n_img_45d_name = orign_img_file+'_45d'
                n_img_45d_path = os.path.join(
                    destination_folder_path,
                    n_img_45d_name + '.pgm'
                )
                Image.open(curr_img_path).rotate(45).save(n_img_45d_path)

                n_names.append(n_img_45d_name)
                n_labels.append(labels[i])

                # rotate 315 degress
                n_img_315d_name = orign_img_file+'_315d'
                n_img_315d_path = os.path.join(
                    destination_folder_path,
                    n_img_315d_name + '.pgm'
                )
                Image.open(curr_img_path).rotate(315).save(n_img_315d_path)

                n_names.append(n_img_315d_name)
                n_labels.append(labels[i])

                # rotate 135 degress
                n_img_135d_name = orign_img_file+'_135d'
                n_img_135d_path = os.path.join(
                    destination_folder_path,
                    n_img_135d_name + '.pgm'
                )
                Image.open(curr_img_path).rotate(135).save(n_img_135d_path)

                n_names.append(n_img_135d_name)
                n_labels.append(labels[i])

                # rotate 225 degress
                n_img_225d_name = orign_img_file+'_225d'
                n_img_225d_path = os.path.join(
                    destination_folder_path,
                    n_img_225d_name + '.pgm'
                )
                Image.open(curr_img_path).rotate(225).save(n_img_225d_path)

                n_names.append(n_img_225d_name)
                n_labels.append(labels[i])

        with open(os.path.join(destination_folder_path, 'augment_info.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerows(zip(n_names, n_labels))
        return n_names, n_labels

    def download(self):
        if self._check_processed_exists():
            return

        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        url = self.dataset_url
        filename = url.rpartition('/')[2]

        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        file_path = os.path.join(self.root, self.raw_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())

        tf = tarfile.open(file_path)
        folder_path = os.path.join(
            self.root, self.raw_folder, filename.replace('.tar.gz', ''))
        tf.extractall(folder_path)
        tf.close()
        os.unlink(file_path)

    def get_class(self, index):
        for name, i in self.labels_info.items():
            if i == index:
                return name

    def _read_pgm(self, filename, byteorder='>'):
        """Return image data from a raw PGM file as a numpy array.

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
                             dtype='u1' if int(
                                 maxval) < 256 else byteorder+'u2',
                             count=int(width)*int(height),
                             offset=len(header)
                             ).reshape((int(height), int(width)))

    def _parse_infos(self, folder):
        regex = r"(mdb\d{3})\W([FGD])\W(CALC|CIRC|SPIC|MISC|ARCH|ASYM|NORM)\W?([BM])?\W?((\d{1,3})\W(\d{1,3})\W(\d{1,3})|(\*NOTE\W3\*))?"
        file_names = []
        labels = []
        for line in open(os.path.join(folder, 'Info.txt')):
            matches = re.finditer(regex, line)
            for matchNum, match in enumerate(matches, start=1):
                file_names.append(match.group(1))
                labels.append(match.group(3))

        return file_names, labels

    def _check_processed_exists(self):
        return os.path.exists(os.path.join(self.root, self.raw_folder, 'all-mias'))

    def plot_distribution(self):
        from collections import Counter
        import numpy as np
        import matplotlib.pyplot as plt

        letter_counts = Counter(self.raw_labels)
        frequencies = letter_counts.values()
        names = letter_counts.keys()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(names, frequencies, align='center')

        plt.show()


# only for test reasons
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = MIAS('./data', download=False)
    print('Dataset size: {}'.format(len(dataset)))
    img, target = dataset[0]
    print(target)
    print(dataset.get_class(target))
    print(img.shape)
    plt.imshow(img, cmap='gray')
    plt.show()

    dataset.plot_distribution()
