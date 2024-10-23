import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, v2
from torchvision import tv_tensors
import numpy as np
import os
import cv2
from nanopyx import eSRRF, SRRF
from src.dynamics import *

def create_datasets_new(config, create_config, calc_config, normalize=True):
    path = config['path']
    train_percent = create_config['train_percent']
    test_percent = create_config.get('test_percent', 0)

    files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.npz'):
                files.append(file_path)

    train_size = int(train_percent * len(files))
    val_size = len(files) - train_size
    train_files, val_files = torch.utils.data.random_split(files, [train_size, val_size])

    if test_percent > 0:
        test_size = int(test_percent * len(files))
        val_size = val_size - test_size
        train_files, val_files, test_files = torch.utils.data.random_split(files, [train_size, val_size, test_size])

    if normalize:
        mean, std = calculate_mean_and_std(path, train_files, calc_config)
    else:
        mean, std = 0, 1

    train_dataset = BiosensorDataset(train_files, mean, std, config.get('augment', False), config, calc_config)
    val_dataset = BiosensorDataset(val_files, mean, std, False, config, calc_config)

    if test_percent > 0:
        test_dataset = BiosensorDataset(test_files, mean, std, False, config, calc_config)
        return train_dataset, val_dataset, test_dataset
    return train_dataset, val_dataset

def calculate_mean_and_std(path, train_files, calc_config):
    biosensor_length = calc_config.get('biosensor_length', 8)
    mask_scale = calc_config.get('mask_scale', 1)
    input_scale = calc_config.get('input_scale', 1)
    srrf = calc_config.get('SRRF_mode', None) # eSRRF, SRRF, None

    input_size = input_scale * 80
    # Preallocate a tensor of the correct size
    if input_scaling:
        data = torch.empty((len(train_files), biosensor_length, input_size, input_size))
    else:
        data = torch.empty((len(train_files), biosensor_length, 80, 80))

    for i, file in enumerate(train_files):
        data = np.load(path + file)
        well, mask = data['well'], data['im_markers']
        scale, t_0, t_1 = int(data['scale']), int(data['t_0']), int(data['t_1'])

        # Slicer only for the biosensor
        shape, scale, translation = mask.shape, (scale, scale), np.array([t_0, t_1])
        start_cardio = np.abs((min(0, translation[1]), min(0, translation[0])))
        over_reach = (-(shape - np.flipud(scale + translation))).clip(min=0)
        end_cardio = np.flipud(scale) - over_reach
        cardio_scale = (80 / scale[0])
        cardio_slice = (slice(None),
                        slice(int(start_cardio[0] * cardio_scale), int(end_cardio[0] * cardio_scale)),
                        slice(int(start_cardio[1] * cardio_scale), int(end_cardio[1] * cardio_scale)))

        cardio = subsample_cardio(well, biosensor_length)
        cardio = cardio[cardio_slice]

        if cardio.shape[1] != 80 or cardio.shape[2] != 80:
            cardio_pad = ((0, 0), (0, 80 - cardio.shape[0]), (0, 80 - cardio.shape[1]))
            cardio = np.pad(cardio, cardio_pad, mode='constant', constant_values=0)

        if input_scale > 1:
            if srrf == 'eSRRF':
                bio = eSRRF(cardio, input_scale)
            elif srrf == 'SRRF':
                bio = SRRF(cardio, input_scale)
            else:
                input_size = input_scale * 80
                bio = cv2.resize(cardio, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
            
        data[i] = bio

    return data.mean(), data.std()

def subsample_cardio(cardio, subsampled_length):
    original_length = cardio.shape[0]
    indices = np.linspace(0, original_length - 1, subsampled_length + 1, dtype=int)
    return cardio[indices[1:]]

def get_slicer(shape, scale, translation):
    start_mic = np.array((max(translation[1], 0), max(translation[0], 0)))
    end_mic = np.flipud((translation + scale))

    cardio_scale = (80 / scale[0])

    start_cardio = np.abs((min(0, translation[1]), min(0, translation[0])))
    over_reach = (-(shape - np.flipud(scale + translation))).clip(min=0)
    end_cardio = np.flipud(scale) - over_reach

    mic_slice = (slice(start_mic[0], end_mic[0]), slice(start_mic[1], end_mic[1]))
    cardio_slice = (slice(None),
                    slice(int(start_cardio[0] * cardio_scale), int(end_cardio[0] * cardio_scale)),
                    slice(int(start_cardio[1] * cardio_scale), int(end_cardio[1] * cardio_scale)))

    return cardio_slice, mic_slice

def get_padding(scale, cardio_shape, mask_shape):
    cardio_pad = ((0, 0), (0, 80 - cardio_shape[0]), (0, 80 - cardio_shape[1]))
    mask_pad = ((0, scale - mask_shape[0]), (0, scale - mask_shape[1]))

    return cardio_pad, mask_pad

def get_data_from_file(file, bio_len=8):
    data = np.load(file)
    well, mask = data['well'], data['im_markers']
    scale, t_0, t_1 = int(data['scale']), int(data['t_0']), int(data['t_1'])

    slicer = get_slicer(mask.shape, (scale, scale), np.array([t_0, t_1]))

    cardio_max = np.max(well, axis=0)
    cardio = subsample_cardio(well, bio_len)
    cardio = cardio[slicer[0]]
    mask = mask[slicer[1]]

    if cardio.shape[1] != 80 or cardio.shape[2] != 80 or mask.shape[0] != scale or mask.shape[1] != scale:
        cardio_pad, mask_pad = get_padding(scale, cardio.shape, mask.shape)
        cardio = np.pad(cardio, cardio_pad, mode='constant', constant_values=0)
        mask = np.pad(mask, mask_pad, mode='constant', constant_values=0)

    _, blank_areas = cv2.threshold(cardio_max, 0, 255, cv2.THRESH_BINARY)
    blank_areas = cv2.resize(blank_areas, (scale, scale), interpolation=cv2.INTER_NEAREST)

    mask = mask.astype(np.uint16)
    mask = cv2.bitwise_and(mask, mask, mask=blank_areas.astype(np.uint8))

    return cardio, mask

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

class BiosensorDataset(Dataset):
    def __init__(self, files, mean, std, augment, config, calc_config):
        self.path = config['path']
        self.files = files
        self.normalize = Normalize(mean=mean, std=std)
        self.mask_type = config.get('mask_type', bool)
        self.bio_length = calc_config.get('biosensor_length', 8)

        # self.mask_size = calc_config.get('mask_size', 80)
        # self.upscale_mode = calc_config.get('upscale_mode', 'nearest')

        self.mask_scale = calc_config.get('mask_scale', 1)
        self.input_scale = calc_config.get('input_scale', 1)
        self.srrf = calc_config.get('SRRF_mode', None) # eSRRF, SRRF, None
        
        self.tiling = config.get('tiling', False)
        self.tiling_ratio = config.get('tiling_ratio', 4)
        self.overlap_rate = config.get('overlap_rate', 0)

        self.dilation = config.get('dilation', 0)
        
        if augment:
            self.transform = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomRotation(90),
            ])
        else:
            self.transform = None

    def __getitem__(self, index):
        bio, mask = get_data_from_file(self.path + self.files[index], self.bio_length)

        if self.input_scale > 1:
            if self.srrf == 'eSRRF':
                bio = eSRRF(bio, self.input_scale)
            elif self.srrf == 'SRRF':
                bio = SRRF(bio, self.input_scale)
            else:
                input_size = self.input_scale * 80
                bio = cv2.resize(bio, (input_size, input_size), interpolation=cv2.INTER_NEAREST)

        mask_size = self.mask_scale * 80
        mask = cv2.resize(mask, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)

        if self.dilation > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations = self.dilation)

        bio = torch.from_numpy(bio.astype(np.float32))
        mask = torch.from_numpy(mask.astype(self.mask_type))

        bio = self.normalize(bio)
        if self.tiling:
            bio, mask = create_tiles(bio, mask, self.tiling_ratio, self.overlap_rate)
        if self.transform:
            mask = tv_tensors.Mask(mask)
            bio, mask = self.transform(bio, mask)
        return bio, mask
        
    def __len__(self):
        return len(self.files)
    
    # def gradient_mask(self, mask):
    #     interpolated_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(self.mask_size, self.mask_size), mode=self.upscale_mode).squeeze(0).squeeze(0).byte()
    #     mask_np = interpolated_mask.numpy()
    #     device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     if device == 'cuda':
    #         flows, centers = masks_to_flows_gpu(mask_np, device)
    #     else:
    #         flows, centers = masks_to_flows_cpu(mask_np)

    #     flow_x, flow_y = flows
    #     mask_bool = mask_np.astype(bool)
    #     output = np.stack((flow_x, flow_y, mask_bool), axis=-1)

    #     return scaled_flow_sum
