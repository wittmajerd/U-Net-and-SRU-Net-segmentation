import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import os

class BiosensorDataset(Dataset):                                          #bool vagy np.int32
    def __init__(self, path, transform=None, biosensor_length=128, mask_size=80, mask_type='np.int32'):
        self.transform = transform
        self.path = path
        self.length = biosensor_length
        self.mask_size = mask_size
        self.mask_type = mask_type
        
        
    def __getitem__(self, index):
        # If the index is a list of indices
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            biosensor = []
            masks = []
            for i in indices:
                data = np.load(self.path + str(i) + '.npz')
                
                bio = self.uniform_time_dim(torch.from_numpy(data['biosensor']))
                mask = self.uniform_mask(torch.from_numpy(data['mask'].astype(self.mask_type)))
                if self.transform:
                    data = self.transform(bio, mask)
                
                biosensor.append(bio)
                masks.append(mask)
            biosensor = torch.stack(biosensor)
            masks = torch.stack(masks)
            return biosensor, masks
        
        # Only one index
        data = np.load(self.path + str(index) + '.npz')
        bio = self.uniform_time_dim(torch.from_numpy(data['biosensor']))
        mask = self.uniform_mask(torch.from_numpy(data['mask'].astype(self.mask_type)))
        if self.transform:
            data = self.transform(bio, mask)
        return bio, mask
        
    def __len__(self):
        # Ezt lehetne jobban de egyelőre csak az npz fájlok vannak a mappában
        return len(os.listdir(self.path))
    
    def uniform_time_dim(self, biosensor):
        indices = np.linspace(0, biosensor.shape[0] - 1, self.length, dtype=int)
        return biosensor[indices]
    
    def uniform_mask(self, mask):
        return torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(self.mask_size, self.mask_size), mode='nearest').squeeze(0).squeeze(0).byte()


# Az a dataset ami egyszerre az össszes adatot betölti ha esetleg később kellene
import pickle

class BiosensorDatasetAll(Dataset):
    def __init__(self, data_path, transform=None):
        self.transform = transform
        with open(data_path + '/alldata.pkl', 'rb') as in_file:
            data = pickle.load(in_file)
            self.filename = data['old_filename']
            self.biosensor = data['biosensor']
            self.masks = data['masks']
        
    def __getitem__(self, index): 
        if self.transform:
            return self.transform(self.biosensor[index]), self.transform(self.masks[index])
        
    def __len__(self):
        return len(self.data)