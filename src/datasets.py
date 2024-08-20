import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, v2
from torchvision import tv_tensors
import numpy as np
import os
import cv2

def create_datasets(path, train_percent, mask_type, test_percent=0, biosensor_length=8, mask_size=80, augment=False, dilation=0, input_scaling=False, upscale_mode='nearest'):
    files = os.listdir(path)
    train_size = int(train_percent * len(files))
    val_size = len(files) - train_size
    train_files, val_files = torch.utils.data.random_split(files, [train_size, val_size])

    if test_percent>0:
        test_size = int(test_percent * len(files))
        val_size = val_size - test_size
        train_files, val_files, test_files = torch.utils.data.random_split(files, [train_size, val_size, test_size])

    mean, std = calculate_mean_and_std(path, train_files, biosensor_length, mask_size, input_scaling, upscale_mode)

    train_dataset = BiosensorDataset(path, train_files, mean, std, mask_type, biosensor_length=biosensor_length, mask_size=mask_size, augment=augment, dilation=dilation, input_scaling=input_scaling, upscale_mode=upscale_mode)
    val_dataset = BiosensorDataset(path, val_files, mean, std, mask_type, biosensor_length=biosensor_length, mask_size=mask_size, augment=False, dilation=dilation, input_scaling=input_scaling, upscale_mode=upscale_mode)
    if test_percent>0:
        test_dataset = BiosensorDataset(path, test_files, mean, std, mask_type, biosensor_length=biosensor_length, mask_size=mask_size, augment=False, dilation=dilation, input_scaling=input_scaling, upscale_mode=upscale)
        return train_dataset, val_dataset, test_dataset
    return train_dataset, val_dataset

# Gives linearly spaced indices for the biosensor subsampling
# n is the length of the original biosensor
# length is the length of the subsampled biosensor
# The indices are 1-indexed because the first frame would be empty so we skip it
def lin_indices(original_length, subsampled_length):
    indices = np.linspace(0, original_length - 1, subsampled_length + 1, dtype=int)
    return indices[1:]

def calculate_mean_and_std(path, train_files, biosensor_length=16, mask_size=80, input_scaling=False, upscale_mode='nearest'):
    # Preallocate a tensor of the correct size
    if input_scaling:
        data = torch.empty((len(train_files), biosensor_length, mask_size, mask_size))
    else:
        data = torch.empty((len(train_files), biosensor_length, 80, 80))

    for i, file in enumerate(train_files):
        loaded_data = np.load(path + file)
        # Get the biosensor data with the correct length
        biosensor = torch.from_numpy(loaded_data['biosensor'].astype(np.float32))
        indices = lin_indices(biosensor.shape[0], biosensor_length)
        bio = biosensor[indices]
        if input_scaling:
            bio = torch.nn.functional.interpolate(bio.unsqueeze(0), size=(mask_size, mask_size), mode=upscale_mode).squeeze(0)
        # Fill the preallocated tensor
        data[i] = bio

    return data.mean(), data.std()


class BiosensorDataset(Dataset):
    def __init__(self, path, files, mean, std, mask_type, biosensor_length=128, mask_size=80, augment=False, dilation=0, input_scaling=False, upscale_mode='nearest'):
        self.path = path
        self.files = files
        self.normalize = Normalize(mean=mean, std=std)
        self.mask_type = mask_type
        self.length = biosensor_length
        self.mask_size = mask_size
        self.dilation = dilation
        self.input_scaling = input_scaling
        self.upscale_mode = upscale_mode
        
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
        bio = self.uniform_biosensor(torch.from_numpy(data['biosensor'].astype(np.float32)))
        mask = self.uniform_mask(torch.from_numpy(data['mask'].astype(self.mask_type)), data['cell_centers'])
        bio = self.normalize(bio)
        if self.transform:
            mask = tv_tensors.Mask(mask)
            bio, mask = self.transform(bio, mask)
        return bio, mask
        
    def __len__(self):
        return len(self.files)
    
    def uniform_biosensor(self, biosensor):
        indices = lin_indices(biosensor.shape[0], self.length)
        if self.input_scaling == False:
            return biosensor[indices]
        downsampled_bio = biosensor[indices]
        upscaled = torch.nn.functional.interpolate(downsampled_bio.unsqueeze(0), size=(self.mask_size, self.mask_size), mode=self.upscale_mode).squeeze(0)
        return upscaled
    
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
        
        if self.dilation > 0:
            # Convert the mask to numpy array for dilation
            mask_np = interpolated_mask.numpy()
            # Define the structuring element for dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilation, self.dilation))
            # Perform dilation
            dilated_mask = cv2.dilate(mask_np, kernel, iterations = 1)
            # Convert the dilated mask back to tensor
            dilated_mask = torch.from_numpy(dilated_mask)
            return dilated_mask
        
        return interpolated_mask
