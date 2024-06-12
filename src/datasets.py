import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, v2
from torchvision import tv_tensors
import numpy as np
import os

def create_datasets(path, train_percent, mask_type, test_percent=0, biosensor_length=8, mask_size=80, augment=False):
    files = os.listdir(path)
    train_size = int(train_percent * len(files))
    val_size = len(files) - train_size
    train_files, val_files = torch.utils.data.random_split(files, [train_size, val_size])

    if test_percent>0:
        test_size = int(test_percent * len(files))
        val_size = val_size - test_size
        train_files, val_files, test_files = torch.utils.data.random_split(files, [train_size, val_size, test_size])

    mean, std = calculate_mean_and_std(path, train_files, biosensor_length)

    train_dataset = BiosensorDataset(path, train_files, mean, std, mask_type, biosensor_length, mask_size, augment)
    val_dataset = BiosensorDataset(path, val_files, mean, std, mask_type, biosensor_length, mask_size)
    if test_percent>0:
        test_dataset = BiosensorDataset(path, test_files, mean, std, mask_type, biosensor_length, mask_size)
        return train_dataset, val_dataset, test_dataset
    return train_dataset, val_dataset

# Gives a linearly spaced indices array of length length for the biosensor subsampling
# n is the length of the original biosensor
# length is the length of the subsampled biosensor
# The indices are 1-indexed because the first frame would be empty so we skip it
def lin_indices(n, length):
    indices = np.linspace(0, n - 1, length + 1, dtype=int)
    return indices[1:]

def calculate_mean_and_std(path, train_files, biosensor_length=16):
    # Preallocate a tensor of the correct size
    data = torch.empty((len(train_files), biosensor_length, 80, 80))

    for i, file in enumerate(train_files):
        loaded_data = np.load(path + file)
        # Get the biosensor data with the correct length
        biosensor = torch.from_numpy(loaded_data['biosensor'].astype(np.float32))
        # indices = np.linspace(0, biosensor.shape[0] - 1, biosensor_length, dtype=int)
        indices = lin_indices(biosensor.shape[0], biosensor_length)
        bio = biosensor[indices]
        # Fill the preallocated tensor
        data[i] = bio

    return data.mean(), data.std()


class BiosensorDataset(Dataset):
    def __init__(self, path, files, mean, std, mask_type, biosensor_length=128, mask_size=80, augment=False):
        self.path = path
        self.files = files
        self.normalize = Normalize(mean=mean, std=std)
        self.mask_type = mask_type
        self.length = biosensor_length
        self.mask_size = mask_size
        if augment:
            self.transform = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(90),
            ])
        else:
            self.transform = None

    def __getitem__(self, index):
        data = np.load(self.path + self.files[index])
        bio = self.uniform_time_dim(torch.from_numpy(data['biosensor'].astype(np.float32)))
        mask = self.uniform_mask(torch.from_numpy(data['mask'].astype(self.mask_type)), data['cell_centers'])
        bio = self.normalize(bio)
        if self.transform:
            mask = tv_tensors.Mask(mask)
            bio, mask = self.transform(bio, mask)
        return bio, mask
        
    def __len__(self):
        return len(self.files)
    
    def uniform_time_dim(self, biosensor):
        # indices = np.linspace(0, biosensor.shape[0] - 1, self.length, dtype=int)
        indices = lin_indices(biosensor.shape[0], self.length)
        return biosensor[indices]
    
    def uniform_mask(self, mask, centers):
        interpolated_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(self.mask_size, self.mask_size), mode='nearest').squeeze(0).squeeze(0).byte()
        
        x_scale = mask.shape[0] / self.mask_size
        y_scale = mask.shape[1] / self.mask_size
        
        scaled_centers = centers / [x_scale, y_scale]

        # Add the cell centers to the mask
        indices = np.transpose(scaled_centers.astype(int))
        if self.mask_type == bool:
            interpolated_mask[indices[0], indices[1]] = True
        else:
            interpolated_mask[indices[0], indices[1]] = 255
        
        return interpolated_mask
