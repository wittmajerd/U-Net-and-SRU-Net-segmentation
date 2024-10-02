import numpy as np
import torch
import skimage.measure
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.evaluate import evaluate


def evaluate_after_training(model, val_loader, test_loader, device):
    val_dice_score, val_detection_rate = evaluate(model, val_loader, device)
    dice_score, detection_rate = evaluate(model, test_loader, device)
    print(f'Validation dice score: {val_dice_score}, Detection rate: {val_detection_rate}')
    print(f'Test dice score: {dice_score}, Detection rate: {detection_rate}')

def plot_results(bio, mask, prediction, binary_prediction):
    plt.figure(figsize=(30, 10))

    bio = bio.squeeze().cpu().detach().numpy()
    mask = mask.squeeze().cpu().detach().numpy()
    prediction = prediction.squeeze().cpu().detach().numpy()
    binary_prediction = binary_prediction.squeeze().cpu().detach().numpy()

    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    colored_mask[mask == 1] = [1, 0, 0, 1]
    colored_mask[mask == 0] = [0, 0, 0, 0]

    colored_prediction = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    colored_prediction[binary_prediction == 1] = [0, 0, 1, 1]
    colored_prediction[binary_prediction == 0] = [0, 0, 0, 0]

    plt.subplot(1, 3, 1)
    plt.imshow(bio, cmap='gray')
    plt.imshow(colored_mask, alpha=0.6)
    plt.title('Biosensor with mask')
    
    plt.subplot(1, 3, 2)
    plt.imshow(prediction, cmap='gray')
    plt.imshow(colored_prediction, alpha=0.6)
    plt.title('Prediction with the binary')

    intercection = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    intercection[(mask == 1) & (binary_prediction == 1)] = [0, 1, 0, 1]

    plt.subplot(1, 3, 3)
    # plt.imshow(bio, cmap='gray')
    plt.imshow(colored_mask)
    plt.imshow(colored_prediction)
    plt.imshow(intercection)
    plt.title('Label and Prediction overlap')
    
    red_patch = mpatches.Patch(color=[1, 0, 0, 1], label='Mask')
    blue_patch = mpatches.Patch(color=[0, 0, 1, 1], label='Prediction')
    green_patch = mpatches.Patch(color=[0, 1, 0, 1], label='Overlap')

    plt.legend(handles=[red_patch, blue_patch, green_patch], loc='upper right', bbox_to_anchor=(1.5, 1))
    
    plt.show()

def plot_loader_data(loader, title):
    for batch_idx, (data, labels) in enumerate(loader):
        # Move the data and labels to the CPU
        data = data.cpu().numpy()
        labels = labels.cpu().numpy()

        if batch_idx == 1:
            break

        # Plot each image in the batch
        for i in range(len(data)):
            index = (batch_idx * len(data) + i + 1)

            plt.figure(figsize=(20, 10))

            # Plot the input image
            plt.subplot(1, 3, 1)
            plt.imshow(data[i][-1], cmap='gray')
            plt.title(f'{title} - Image {index} ')

            # Plot the label
            plt.subplot(1, 3, 2)
            plt.imshow(labels[i], cmap='gray')
            plt.title(f'{title} - Label {index}')

            plt.subplot(1, 3, 3)
            plt.imshow(data[i][-1], cmap='gray')
            plt.imshow(labels[i], cmap='Reds', alpha=0.25)

            plt.show()

def plot_loader_tiles_data(loader, title):
    for batch_idx, (data, labels) in enumerate(loader):
        # Move the data and labels to the CPU
        # data = data.cpu().numpy()
        # labels = labels.cpu().numpy()

        if batch_idx == 1:
            break

        for i in range(len(data)):
            plot_tiles(data[i], labels[i])

def create_tiles(bio, mask, ratio):
    ch, bh, bw = bio.shape
    mh, mw = mask.shape
    bio_size = bh // ratio
    mask_size = mh // ratio

    bio_tiles = bio.reshape(ch, ratio, bio_size, ratio, bio_size).permute(1, 3, 0, 2, 4).reshape(ratio * ratio, ch, bio_size, bio_size)
    mask_tiles = mask.reshape(ratio, mask_size, ratio, mask_size).permute(0, 2, 1, 3).reshape(ratio * ratio, mask_size, mask_size)
    print(bio_tiles.shape, mask_tiles.shape)

    return bio_tiles, mask_tiles

def merge_tiles(tiles):
    n, h, w = tiles.shape
    ratio = int(np.sqrt(n))
    merged = tiles.reshape(ratio, ratio, h, w).permute(0, 2, 1, 3).reshape(ratio * h, ratio * w)
    return merged

def plot_tiles(bio_tiles, mask_tiles):
    n, ch, h, w = bio_tiles.shape
    ratio = int(np.sqrt(n))

    fig, ax = plt.subplots(ratio, ratio, figsize=(8, 8))
    for i in range(ratio):
        for j in range(ratio):
            ax[i, j].imshow(bio_tiles[j + i * ratio, -1], cmap='gray')
            ax[i, j].imshow(mask_tiles[j + i * ratio], alpha=0.6)
            ax[i, j].axis('off')
    plt.tight_layout()
    plt.show()


def cell_detection_skimage(model, loader, device, threshold=0.5):
    model.eval()  # Set the model to evaluation mode

    total_cells = 0
    detected_cells = 0

    with torch.no_grad():  # Disable gradient calculation
        for data, labels in loader:
            # Move the data and labels to the device
            data = data.to(device)
            labels = labels.to(device)

            # Get the predictions
            predictions = model(data)

            # Move the predictions and labels to the CPU and convert them to numpy arrays
            predictions = predictions.cpu().detach().numpy()
            binary_predictions = (predictions > threshold).astype(np.uint8)

            labels = labels.cpu().numpy()

            for i in range(len(data)):
                # Label the binary prediction and count the number of cells
                _, num_cells_pred = skimage.measure.label(binary_predictions[i], return_num=True) #connectivity=2
                _, num_cells_label = skimage.measure.label(labels[i], return_num=True)

                total_cells += num_cells_label
                detected_cells += num_cells_pred

    cell_detection_rate = detected_cells / total_cells if total_cells > 0 else 0

    return cell_detection_rate, total_cells, detected_cells

def cell_detection_scipy(model, loader, device, threshold=0.5):
    model.eval()  # Set the model to evaluation mode

    total_cells = 0
    detected_cells = 0

    with torch.no_grad():  # Disable gradient calculation
        for data, labels in loader:
            # Move the data and labels to the device
            data = data.to(device)
            labels = labels.to(device)

            # Get the predictions
            predictions = model(data)
            # predictions = predictions.cpu().detach().numpy()
            # Move the predictions and labels to the CPU and convert them to numpy arrays
            # binary_predictions = (predictions > threshold).astype(np.uint8)
            binary_predictions = (torch.nn.functional.sigmoid(predictions) > threshold)
            binary_predictions = binary_predictions.cpu().detach().numpy()
            labels = labels.cpu().numpy()

            structure = np.ones((3, 3))

            for i in range(len(data)):
                # Label the binary prediction and count the number of cells
                _, num_cells_pred = scipy.ndimage.label(np.squeeze(binary_predictions[i]), structure=structure)
                _, num_cells_label = scipy.ndimage.label(labels[i], structure=structure)

                total_cells += num_cells_label
                detected_cells += num_cells_pred

    cell_detection_rate = detected_cells / total_cells if total_cells > 0 else 0

    return cell_detection_rate, total_cells, detected_cells

def pos_pixels(model, val_loader, device, threshold=0.5):
    model.eval()  # Set the model to evaluation mode

    total_cells = 0
    detected_cells = 0

    with torch.no_grad():  # Disable gradient calculation
        for data, labels in val_loader:
            # Move the data and labels to the device
            data = data.to(device)
            labels = labels.to(device)

            # Get the predictions
            predictions = model(data)

            # Move the predictions and labels to the CPU and convert them to numpy arrays
            predictions = predictions.cpu().detach().numpy()
            binary_predictions = (predictions > threshold).astype(np.uint8)

            labels = labels.cpu().numpy()

            total_cells += np.sum(labels)
            detected_cells += np.sum(binary_predictions)

    return total_cells, detected_cells