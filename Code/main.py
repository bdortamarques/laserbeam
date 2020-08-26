import laserbeam
from glob import glob
from os.path import basename
import os.path as path
import pandas as pd
from os.path import join
import random

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from imageio import imread

import os
from os.path import join
from os.path import exists

from datetime import datetime
from glob import glob
from tqdm import tqdm
from tqdm import trange
import time

np.random.seed(5555)
random.seed(5555)

def load_dataframe(root_dir):
    data = {}
    data['image_1']=[]
    data['image_2']=[]
    data['theta']=[]
    data['phi']=[]
    for image in glob( join(root_dir, '*.png')):
        if "image_1" in image:
            theta = image.split('_')[3][1:]
            phi = image.split('_')[4][:-4][3:]
            data['image_1'].append(path.basename(image))
            data['image_2'].append(path.basename(image).replace('image_1','image_2'))
            data['theta'].append(int(theta))        
            data['phi'].append(int(phi))     
    dataframe = pd.DataFrame(data=data)
    return dataframe



def create_dataloaders(dataset_path):
    

    df = load_dataframe(dataset_path).sample(frac=1.0, random_state=5555)
    train=df.sample(frac=0.75) #random state is a seed value
    df.drop(train.index, inplace=True)
    val = df.sample(frac=0.60)
    test = df.drop(val.index)
    
    datasets = {'train': laserbeam.LaserDataset(dataset_path, train),
                'val' : laserbeam.LaserDataset(dataset_path, val), 
                'test': laserbeam.LaserDataset(dataset_path, test)}
    
    dataloaders = {}
    for phase in datasets:
        dataloaders[phase] = DataLoader( datasets[phase], batch_size=16, shuffle=True, num_workers=1, pin_memory=True)
    return dataloaders, datasets

dataloaders, datasets = create_dataloaders('D:\\laserBeam-2020\\Dataset\\ml_v0')
model = laserbeam.LaserModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model_path = 'D:\\laserBeam-2020\\Models\\laserbean_v0'
if not exists(model_path):
    os.mkdir(model_path)

current_epoch = 0

history = {'train_loss' : [],
            'val_loss' : [] }

device = torch.device("cuda")
model.to(device)


def save_checkpoint():
    checkpoint_dict = {
        'epoch': current_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion': criterion,
        'history' : history }
        
    checkpoint_path = join(model_path, "checkpoint")
    if not exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    torch.save(checkpoint_dict, join(checkpoint_path,"epoch{:03d}_".format(current_epoch) + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+".pth"))

def load_checkpoint(path=None):
    global current_epoch
    global optimizer
    global model
    global history
    global criterion

    if not path: #use the last checkpoint
        path = list(glob(join(model_path, 'checkpoint', '*')))
        path.sort()
        path = path[-1]

    
    checkpoint_dict = torch.load(path)
    model.load_state_dict(checkpoint_dict['model_state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    criterion = checkpoint_dict['criterion']
    history = checkpoint_dict['history']
    current_epoch = checkpoint_dict['epoch']
    print("Loaded checkpoint {} at epoch {:2d}, with val_loss: {}".format(
        os.path.basename(path), checkpoint_dict['epoch'], history['val_loss'][-1]))


def train_model(epochs = 2, patience = 5):
    
    for epoch in range(current_epoch, epochs):
        if len(history['val_loss']) > patience:
            previous_loss = history['val_loss'][-(patience+1):-1] 
            last_loss = history['val_loss'][-1]
            if any( loss < last_loss for loss in previous_loss):
                best_loss_index = np.argmin(history['val_loss'])
                if epoch - best_loss_index >= patience :
                    print("Loss did not improved (patience {})| best epoch: {}".format(patience, best_loss_index))
                    break
            
        train_loop()

        
def train_loop():
    global current_epoch
    model.train() # Training
    train_loss = 0.0
    train_progress = tqdm(dataloaders['train'], desc='|Epoch {}| Training Starting'.format(current_epoch))
    for i, batch in enumerate(train_progress):
        
        batch_images = batch['image'].to(device)
        batch_labels =  batch['angles'].to(device)
        
        optimizer.zero_grad() # zero the parameter gradients
        # forward + backward + optimize
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_progress.set_description("|Epoch {}| Training - Batch {} | Loss: {} | Status".format(current_epoch, i+1, train_loss/ (i+1)))
    train_loss = train_loss/(i+1.0)
    history['train_loss'].append(train_loss)
    evaluate(phase='val') # Validation
    current_epoch += 1
    save_checkpoint()

def evaluate(phase='test'):
    if phase == 'val':
        phase_description = "Validation"
    else:
        phase_description = str.capitalize(phase)
        eval_history = {'loss' : [],
                        'input' : [],
                        'groundtruth' : [],
                        'output' : []}
    model.eval() # Validation
    with torch.no_grad():
        val_loss = 0.0
        val_progress = tqdm(dataloaders[phase], desc='|Epoch {}| {} Starting'.format(current_epoch, phase_description))
        for i, batch in enumerate(val_progress):
            batch_images = batch['image'].to(device)
            batch_labels =  batch['angles'].to(device)
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item()
            val_progress.set_description("|Epoch {}| {} - Batch {} | Loss: {} | Status".format(current_epoch, phase_description, i+1, val_loss/ (i+1)))
            if phase == 'test':
                eval_history['input'].append(batch_images)
                eval_history['groundtruth'].append(batch_labels)
                eval_history['output'].append(outputs)
        val_loss = val_loss/(i+1.0)
        print('|epoch {:d}| Mean loss: {:3f}'.format(current_epoch, val_loss))
        if phase == 'val':
            history['val_loss'].append(val_loss)
        else:
            eval_history['input'] = torch.cat(eval_history['input']).permute((0,2,3,1))
            eval_history['output'] = torch.cat(eval_history['output'])
            eval_history['groundtruth'] = torch.cat(eval_history['groundtruth'])
            eval_history['loss'].append(val_loss)
            return eval_history


def test_single_example(example, root_dir = 'D:\\laserBeam-2020\\Dataset\\ml_v0'):
    with torch.no_grad():
        image_1 = (torch.Tensor(imread(join(root_dir,example.image_1))) / 255.0)
        image_2 = (torch.Tensor(imread(join(root_dir, example.image_2))) / 255.0)
        image = torch.stack([image_1, image_2]).expand(1,2,224,224)
        image = image.cuda()
        output = model(image)
        return output.cpu()

if __name__ == '__main__':

    if os.path.exists(join(model_path, "checkpoint")):
        #load_checkpoint(join(model_path, "checkpoint", 'epoch069_2020-08-25_14-22-06.pth'))
        load_checkpoint()

    #print(history, current_epoch)
    #train_model(100,20)
    evaluate()


    #example = datasets['test'].samples_dataframe.iloc[500]
    #output = test_single_example(example)
    #print(180.0 * output)
    #print(example)
    
