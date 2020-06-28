from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
from sklearn.model_selection import train_test_split
import numpy as np
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        s_path = 'Caltech101/'+split+'.txt'

        self.label_names = {}
        self.images = []
        self.labels = []
        class_cnt = 0

        for im_path in open(s_path, 'r'):
            im_path = im_path.strip()
            label_name = im_path.split("/")[0]
            if label_name != 'BACKGROUND_Google':
                image = pil_loader(os.path.join(root, im_path))
                self.images.append(image)
                if label_name not in self.label_names:
                    self.label_names[label_name] = class_cnt
                    class_cnt += 1
                self.labels.append(self.label_names[label_name])



    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.images[index], self.labels[index] # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label


    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.labels) # Provide a way to get the length (number of elements) of the dataset
        return length

    def train_val_split(self, val_size=None, random_state=None):
        train_indices, val_indices = train_test_split(np.arange(len(self.labels)),
                                                      test_size=val_size,
                                                      shuffle=False,
                                                      stratify=self.labels)

        return train_indices, val_indices

