import torch.utils.data as data
from six.moves import urllib
import numpy as np
import tarfile
import errno
import gzip
import re
import os

class MIAS(data.Dataset):

  raw_folder = 'raw'
  processed_folder = 'processed'
  dataset_file = 'dataset.pt'
  urls = {
    'dataset' : 'http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz'
  }

  def __init__(self, root, transform=None, target_transform=None, download=True):
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    if download:
      self.download()

    # TODO proces files to build an unique dataset file
    self.parse_data()

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    """
    Args:
      index (int): Index
    Returns:
      tuple: (image, target)
      where target is index of the target class and info contains
    """

    path, target = self.samples[index]
    sample = self._read_pgm(path)
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)

    return sample, target

  def parse_data(self):
    url = self.urls['dataset']
    filename = url.rpartition('/')[2]
    folder_path = os.path.join(self.root, self.raw_folder, filename.replace('.tar.gz', ''))
    file_names, labels = self._parse_infos(folder_path)
    self.file_paths = [ os.path.join(folder_path, file_name) for file_name in file_names ]
    self.labels_info = { v:i for i,v in enumerate( list({ i for i in labels }) ) }
    self.targets = [ self.labels_info[label] for label in labels ]
    self.samples = list(zip(self.file_paths, self.targets))

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

    url = self.urls['dataset']
    filename = url.rpartition('/')[2]

    print('Downloading ' + url)
    data = urllib.request.urlopen(url)
    file_path = os.path.join(self.root, self.raw_folder, filename)
    with open(file_path, 'wb') as f:
      f.write(data.read())

    tf = tarfile.open(file_path)
    folder_path = os.path.join(self.root, self.raw_folder, filename.replace('.tar.gz', ''))
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
                dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                count=int(width)*int(height),
                offset=len(header)
                ).reshape(( int(height), int(width) ))

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
    return os.path.exists(os.path.join(self.root, self.processed_folder, self.dataset_file))

# only for test reasons
if __name__ == "__main__":
  import matplotlib.pyplot as plt

  dataset = MIAS('./data', download=False)
  img, target = dataset[0]
  print(target)
  print(dataset.get_class(target))
  print(img.shape)
  plt.imshow(img, cmap='gray')
  plt.show()