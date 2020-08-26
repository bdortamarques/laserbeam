import torch

from torch.utils.data import Dataset
from glob import glob
from os.path import join
from imageio import imread
import pickle
import matplotlib.pyplot as plt
import numpy as np


class LaserDataset(Dataset):
    """Laser Beam dataset."""


    def __init__(self, root_dir, samples_dataframe):
        """
        Args:
            root_dir (string): Directory with all laser images folders.
        """
        self.root_dir = root_dir
        self.samples_dataframe = samples_dataframe
        self.num_samples = len(samples_dataframe)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):

        sample = self.samples_dataframe.iloc[index]

        image_1 = (torch.Tensor(imread(join(self.root_dir,sample.image_1))) / 255.0)
        image_2 = (torch.Tensor(imread(join(self.root_dir, sample.image_2))) / 255.0)

        image = torch.stack([image_1, image_2])



        label = torch.Tensor([sample.theta/180.0, sample.phi/180.0])
        sample = {'image': image, 'angles': label}

        return sample