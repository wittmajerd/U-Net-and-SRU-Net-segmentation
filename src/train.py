from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from torchinfo import summary

import wandb
from src.dice_score import dice_loss
from src.evaluate import evaluate, predict_image

class_labels = {0: 'background', 1: 'cell'}

def train_model(
    model: nn.Module,
    project_name,
    model_name,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    epochs: int = 5,
    amp: bool = False,
    checkpoint_dir = Path('checkpoints'),
    wandb_logging: bool = False,
    tile_ratio: int = 1,
):
    assert model.n_classes == 1, 'Can only train binary classification model with this function'

    if wandb_logging:
        # (Initialize logging)
        experiment = wandb.init(
            project=project_name, 
            resume='allow', 
            anonymous='must',
            name=model_name,)
        experiment.config.update({
            'epochs': epochs,
            'batch_size': train_loader.batch_size,
            'learning_rate': learning_rate,
            'bio_len': train_loader.dataset.length,
            'amp': amp,
            'tile_ratio': tile_ratio,
            'trainable_params': summary(model).trainable_params,
        })

    print(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {train_loader.batch_size}
        Learning rate:   {learning_rate}
        Training size:   {len(train_loader.dataset)}
        Validation size: {len(val_loader.dataset)}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    grad_clipping = 1.0
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp) # Needed because of autocasting
    criterion = nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for i, (batch) in enumerate(train_loader):
                images, true_masks = batch

                if tile_ratio > 1:
                    # Reshape images and masks to merge tile dimension with batch dimension
                    batch_size, num_tiles, channels, height, width = images.shape
                    images = images.view(batch_size * num_tiles, channels, height, width)
                    batch_size, num_tiles, height, width = true_masks.shape
                    true_masks = true_masks.view(batch_size * num_tiles, height, width)

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                # Clear gradients left by previous batch
                optimizer.zero_grad(set_to_none=True) # For reduced memory footprint

                # Forward pass
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred_logits = model(images)
                    loss = criterion(masks_pred_logits, true_masks.unsqueeze(1))

                    # Predictions with probabilities
                    masks_pred = F.sigmoid(masks_pred_logits).squeeze(dim=1)
                    loss += dice_loss(
                        masks_pred.float(),
                        torch.squeeze(true_masks, dim=1).float(),
                        multiclass=False
                    )

                grad_scaler.scale(loss).backward() # Populate gradients
                nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)
                grad_scaler.step(optimizer) # Do optimization step
                grad_scaler.update()

                # Update statistics
                pbar.update(train_loader.batch_size)
                global_step += 1
                epoch_loss += loss.item()
                if wandb_logging:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Log statistics
                if (i+1) == len(train_loader):
                    val_score, detection_rate = evaluate(model, val_loader, device)
                    print(f'Validation Dice score: {val_score}, Detection rate: {detection_rate}')

                    # Reduce lr when validation does not get better
                    scheduler.step(val_score)

                    ixs = torch.nonzero(true_masks)[:, 0]
                    image_ix = ixs[0].item() if ixs.size(0) > 0 else 0
                    predicted_mask = (masks_pred[image_ix] > 0.5).cpu()
                    ground_truth = torch.squeeze(true_masks[image_ix], dim=0).cpu()

                    if wandb_logging:
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'image': wandb.Image(
                                to_pil_image(images[image_ix][-1]), 
                                masks={
                                    'prediction': {'mask_data': predicted_mask.numpy(), 'class_labels': class_labels},
                                    'ground_truth': {'mask_data': ground_truth.numpy(), 'class_labels': class_labels}
                                },
                            ),
                            'masks': {
                                'pred': wandb.Image(predicted_mask.float()),
                                'true': wandb.Image(ground_truth.float()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            'detection_rate': detection_rate,
                        })

        # Save checkpoint
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        save_model(model, epoch, optimizer.param_groups[0]['lr'], checkpoint_dir)
        print(f'Checkpoint {epoch} saved!')

    if wandb_logging:
        wandb.finish()


def save_model(model: nn.Module, epoch: int, lr: float, dir):
    dir = Path(dir)
    state_dict = model.state_dict()
    state_dict['learning_rate'] = lr
    torch.save(state_dict, str(dir / f'checkpoint_epoch{epoch}.pth'))
