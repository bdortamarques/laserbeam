import numpy as np
import random
import os
from os.path import exists, basename, join

from glob import glob

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd
from imageio import imread



from tqdm import tqdm
from tqdm import trange

from datetime import datetime

from .laserDataset import LaserDataset
from .laserModel import LaserModel


class Trainer():

    def __init__(self):
     
        np.random.seed(5555)
        random.seed(5555)

        self.dataloaders, self.datasets = self.create_dataloaders('D:\\laserBeam-2020\\Dataset\\ml_v0')

        self.model_path = 'D:\\laserBeam-2020\\Models\\laserbean_v0'
        if not exists(self.model_path):
            os.mkdir(self.model_path)

        self.setup()

    def setup(self):
        
        self.model = LaserModel()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.current_epoch = 0

        self.history = {'train_loss' : [],
                    'val_loss' : [] }

        self.device = torch.device("cuda")
        self.model.to(self.device)



    def load_dataframe(self, root_dir):
        data = {}
        data['image_1'] = []
        data['image_2'] = []
        data['theta'] = []
        data['phi'] = []
        for image in glob(join(root_dir, '*.png')):
            if "image_1" in image:

                theta = image.split('_')[3][1:]
                phi = image.split('_')[4][:-4][3:]
                data['image_1'].append(basename(image))
                data['image_2'].append(basename(image).replace('image_1', 'image_2'))
                data['theta'].append(int(theta))
                data['phi'].append(int(phi))
        self.dataframe = pd.DataFrame(data=data)
        return self.dataframe


    def create_dataloaders(self, dataset_path):


        df = self.load_dataframe(dataset_path).sample(frac=1.0, random_state=5555)
        train = df.sample(frac=0.75) #random state is a seed value
        df.drop(train.index, inplace=True)
        val = df.sample(frac=0.60)
        test = df.drop(val.index)

        self.datasets = {'train': LaserDataset(dataset_path, train),
                    'val' : LaserDataset(dataset_path, val), 
                    'test': LaserDataset(dataset_path, test)}

        self.dataloaders = {}
        for phase in self.datasets:
            self.dataloaders[phase] = DataLoader( self.datasets[phase], batch_size=16, shuffle=True, num_workers=1, pin_memory=True)
        return self.dataloaders, self.datasets


    def save_checkpoint(self):
        checkpoint_dict = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion': self.criterion,
            'history' : self.history }

        checkpoint_path = join(self.model_path, "checkpoint")
        if not exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        torch.save(checkpoint_dict, join(checkpoint_path, "epoch{:03d}_".format(self.current_epoch) + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".pth"))

    def load_checkpoint(self, path=None):

        if not path: #use the last checkpoint
            path = list(glob(join(self.model_path, 'checkpoint', '*')))
            path.sort()
            path = path[-1]


        checkpoint_dict = torch.load(path)
        self.model.load_state_dict(checkpoint_dict['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        self.criterion = checkpoint_dict['criterion']
        self.history = checkpoint_dict['history']
        self.current_epoch = checkpoint_dict['epoch']
        print("Loaded checkpoint {} at epoch {:2d}, with val_loss: {}".format(
            os.path.basename(path), checkpoint_dict['epoch'], self.history['val_loss'][-1]))


    def train_model(self, epochs = 2, patience = 5):

        for epoch in range(self.current_epoch, epochs):
            if len(self.history['val_loss']) > patience:
                previous_loss = self.history['val_loss'][-(patience + 1): -1] 
                last_loss = self.history['val_loss'][-1]
                if any(loss < last_loss for loss in previous_loss):
                    best_loss_index = np.argmin(self.history['val_loss'])
                    if epoch - best_loss_index >= patience:
                        print("Loss did not improved (patience {})| best epoch: {}".format(patience, best_loss_index))
                        break
                    
            self.train_loop()


    def train_loop(self):

        self.model.train() # Training
        train_loss = 0.0
        train_progress = tqdm(self.dataloaders['train'], desc='|Epoch {}| Training Starting'.format(self.current_epoch))
        for i, batch in enumerate(train_progress):

            batch_images = batch['image'].to(self.device)
            batch_labels =  batch['angles'].to(self.device)

            self.optimizer.zero_grad() # zero the parameter gradients
            # forward + backward + optimize
            outputs = self.model(batch_images)
            loss = self.criterion(outputs, batch_labels)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            train_progress.set_description("|Epoch {}| Training - Batch {} | Loss: {} | Status".format(self.current_epoch, i+1, train_loss/ (i+1)))
        train_loss = train_loss/(i + 1.0)
        self.history['train_loss'].append(train_loss)
        self.evaluate(phase='val') # Validation
        self.current_epoch += 1
        self.save_checkpoint()

    def evaluate(self, phase='test'):
        if phase == 'val':
            phase_description = "Validation"
        else:
            phase_description = str.capitalize(phase)
            eval_history = {'loss' : [],
                            'input' : [],
                            'groundtruth' : [],
                            'output' : []}
        self.model.eval() # Validation
        with torch.no_grad():
            val_loss = 0.0
            val_progress = tqdm(self.dataloaders[phase], desc='|Epoch {}| {} Starting'.format(self.current_epoch, phase_description))
            for i, batch in enumerate(val_progress):
                batch_images = batch['image'].to(self.device)
                batch_labels =  batch['angles'].to(self.device)
                outputs = self.model(batch_images)
                loss = self.criterion(outputs, batch_labels)
                val_loss += loss.item()
                val_progress.set_description("|Epoch {}| {} - Batch {} | Loss: {} | Status".format(self.current_epoch, phase_description, i+1, val_loss/ (i+1)))
                if phase == 'test':
                    eval_history['input'].append(batch_images)
                    eval_history['groundtruth'].append(batch_labels)
                    eval_history['output'].append(outputs)
            val_loss = val_loss/(i+1.0)
            print('|epoch {:d}| Mean loss: {:3f}'.format(self.current_epoch, val_loss))
            if phase == 'val':
                self.history['val_loss'].append(val_loss)
            else:
                eval_history['input'] = torch.cat(eval_history['input']).permute((0, 2, 3, 1))
                eval_history['output'] = torch.cat(eval_history['output'])
                eval_history['groundtruth'] = torch.cat(eval_history['groundtruth'])
                eval_history['loss'].append(val_loss)
                return eval_history


    def test_single_example(self, example, root_dir = 'D:\\laserBeam-2020\\Dataset\\ml_v0'):
        with torch.no_grad():
            image_1 = (torch.Tensor(imread(join(root_dir, example.image_1))) / 255.0)
            image_2 = (torch.Tensor(imread(join(root_dir, example.image_2))) / 255.0)
            image = torch.stack([image_1, image_2]).expand(1, 2, 224, 224)
            image = image.cuda()
            output = self.model(image)
            return output.cpu()
