import os
import pickle
import torch
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from crnn import CRNN


class FixHeightResize(object):
    """
    Scale images to fixed height
    """

    def __init__(self, height=32, minwidth=100):
        self.height = height
        self.minwidth = minwidth

    # img is an instance of PIL.Image
    def __call__(self, img):
        w, h = img.size
        width = max(int(w * self.height / h), self.minwidth)
        return img.resize((width, self.height), Image.ANTIALIAS)


class SingleRatioImage(Dataset):
    """
    dataset to store images of the same ratio，(torch.utils.data.Dataset)

    Args:
        root (string): Root directory of images
        training (bool, optional): If True, train the model, otherwise test it (default: True)
        data_size (int, optional): size of data to load (default: All data)
    """

    def __init__(self, root, ratio, data_list):
        super().__init__()
        self.data_list = data_list

        # image resize + grayscale + transform to tensor
        self.transform = transforms.Compose((
            transforms.Grayscale(),
            transforms.Resize((32, 32 * ratio), Image.ANTIALIAS),
            transforms.ToTensor()
        ))

        self.folder = os.path.join(root, str(ratio))

    def __len__(self, ):
        return len(self.data_list)

    def __getitem__(self, idx):
        name, label = self.data_list[idx]
        img = self.transform(Image.open(os.path.join(self.folder, name)))
        return img, label


class LoadIter(object):
    """
    """

    def __init__(self, loaders):
        self.lengths = [len(x) for x in loaders]
        self.loaders = loaders
        self.iters = [loader.__iter__() for loader in self.loaders]
        self.Ps = [self.lengths[i] / sum(self.lengths)
                   for i in range(len(self.iters))]
        self.avaliable_iters = range(len(self.iters))

    def __iter__(self):
        return self

    def get_data(self, iter_id):
        try:
            return next(self.iters[iter_id])
        except StopIteration:
            self.lengths[iter_id] = 0
            if sum(self.lengths) == 0:
                raise StopIteration
            self.Ps = [self.lengths[i] / sum(self.lengths)
                       for i in range(len(self.iters))]
            return next(self)

    def __next__(self):
        random_id = np.random.choice(self.avaliable_iters, p=self.Ps)
        return self.get_data(random_id)


class Loader(object):
    """
    manager of three datasets，(torch.utils.data.Dataset)

    Args:
        root (string): Root directory of dataset
        batch_size (int): Size of each batch
        training (bool, optional): If True, train the model, otherwise test it (default: True)
        data_size (int, optional): size of data to load (default: All data)
    """

    def __init__(self, root, batch_size, training=True, data_size=None):
        # load data list
        data_list = {2:[], 5:[], 8:[]}
        with open(os.path.join(root, 'label.txt')) as fp:
            for line in fp.readlines()[:data_size]:
                folder, name, label = line.split()[:3]
                data_list[int(folder)].append((name+'.jpg', label))
        img_root = os.path.join(root, 'image')

        # get data loader
        datasets = [SingleRatioImage(img_root, k, data_list[k])
                    for k in data_list]
        del data_list
        self.loaders = []
        for dataset in datasets:
            self.loaders.append(DataLoader(dataset, batch_size=batch_size,
                                           shuffle=training, num_workers=4))
        del datasets

    def __iter__(self):
        return LoadIter(self.loaders)


class LabelTransformer(object):
    """
    encoder and decoder

    Args:
        letters (str): Letters contained in the data
    """

    def __init__(self, letters):
        letters = ''.join(set(letters))
        self.encode_map = {letter: idx+1 for idx, letter in enumerate(letters)}
        self.decode_map = ''.join((' ', letters))

    def encode(self, text):
        if isinstance(text, str):
            length = [len(text)]
            for letter in text:
                if letter not in self.encode_map:
                    result.append(len(self.decode_map))  # letters not in the dict is encoded as len(letters)+2
                else:
                    result.append(self.encode_map[letter])
        else:
            length = []
            result = []
            for word in text:
                length.append(len(word))
                for letter in word:
                    if letter not in self.encode_map:
                        result.append(len(self.decode_map))  # letters not in the dict is encoded as len(letters)+2
                    else:
                        result.append(self.encode_map[letter])
        return torch.IntTensor(result), torch.IntTensor(length)

    def decode(self, text_code):
        result = []
        for code in text_code:
            word = []
            for i in range(len(code)):
                if code[i] != 0 and (i == 0 or code[i] != code[i-1]):
                    word.append(self.decode_map[code[i]])
            result.append(''.join(word))
        return result