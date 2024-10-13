import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, v2
from torchvision import tv_tensors
import numpy as np
import os
import cv2

def create_datasets(config, create_config, calc_config):
    path = config['path']
    train_percent = create_config['train_percent']
    test_percent = create_config.get('test_percent', 0)

    files = os.listdir(path)
    train_size = int(train_percent * len(files))
    val_size = len(files) - train_size
    train_files, val_files = torch.utils.data.random_split(files, [train_size, val_size])

    if test_percent > 0:
        test_size = int(test_percent * len(files))
        val_size = val_size - test_size
        train_files, val_files, test_files = torch.utils.data.random_split(files, [train_size, val_size, test_size])

    mean, std = calculate_mean_and_std(path, train_files, calc_config)

    train_dataset = BiosensorDataset(train_files, mean, std, config.get('augment', False), config, calc_config)
    val_dataset = BiosensorDataset(val_files, mean, std, False, config, calc_config)
    if test_percent>0:
        test_dataset = BiosensorDataset(test_files, mean, std, False, config, calc_config)
        return train_dataset, val_dataset, test_dataset
    return train_dataset, val_dataset

# Gives linearly spaced indices for the biosensor subsampling
# n is the length of the original biosensor
# length is the length of the subsampled biosensor
# The indices are 1-indexed because the first frame would be empty so we skip it
def lin_indices(original_length, subsampled_length):
    indices = np.linspace(0, original_length - 1, subsampled_length + 1, dtype=int)
    return indices[1:]

# Create tiles of the biosensor and mask with the given tile ratio and overlap rate
# Output biosensor shape: (ratio * ratio, ch, size, size)
# Output mask shape: (ratio * ratio, size, size)
# So basically creates ratio * ratio batches of size x size tiles
# If the mask is bigger then the tiles are also bigger at the same rate so SRU models can be trained
# If the overlap rate is positive then the tiles will overlap so there will be more tiles than the ratio * ratio
def create_tiles(bio, mask, ratio, overlap_rate=0):
    ch, bio_h, bio_w = bio.shape
    mask_h, mask_w = mask.shape
    bio_size = bio_h // ratio
    mask_size = mask_h // ratio

    bio_stride = bio_size - int(bio_size * overlap_rate)
    mask_stride = mask_size - int(mask_size * overlap_rate)

    bio_tiles = bio.unfold(1, bio_size, bio_stride).unfold(2, bio_size, bio_stride)
    bio_tiles = bio_tiles.permute(1, 2, 0, 3, 4).reshape(-1, ch, bio_size, bio_size)
    mask_tiles = mask.unfold(0, mask_size, mask_stride).unfold(1, mask_size, mask_stride)
    mask_tiles = mask_tiles.permute(0, 1, 2, 3).reshape(-1, mask_size, mask_size)

    return bio_tiles, mask_tiles

def calculate_mean_and_std(path, train_files, calc_config):
    biosensor_length = calc_config.get('biosensor_length', 8)
    mask_size = calc_config.get('mask_size', 80)
    input_scaling = calc_config.get('input_scaling', False)
    upscale_mode = calc_config.get('upscale_mode', 'nearest')

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
    def __init__(self, files, mean, std, augment, config, calc_config):
        self.path = config['path']
        self.files = files
        self.normalize = Normalize(mean=mean, std=std)
        self.mask_type = config.get('mask_type', bool)
        self.length = calc_config.get('biosensor_length', 8)
        self.mask_size = calc_config.get('mask_size', 80)
        self.dilation = config.get('dilation', 0)
        self.input_scaling = calc_config.get('input_scaling', False)
        self.upscale_mode = calc_config.get('upscale_mode', 'nearest')
        self.noise = config.get('noise', 0.0)
        self.tiling = config.get('tiling', False)
        self.tiling_ratio = config.get('tiling_ratio', 4)
        self.overlap_rate = config.get('overlap_rate', 0)
        
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
            bio = AddGaussianNoise(0., self.noise)(bio)
        if self.tiling:
            bio, mask = create_tiles(bio, mask, self.tiling_ratio, self.overlap_rate)
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
        interpolated_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(self.mask_size, self.mask_size), mode=self.upscale_mode).squeeze(0).squeeze(0).byte()
        
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

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"