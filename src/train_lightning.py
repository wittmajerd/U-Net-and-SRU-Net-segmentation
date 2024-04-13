import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, random_split
from src.unet import UNet
from src.datasets import BiosensorDataset
from torch.nn import Module

class UNetLightningModule(LightningModule):
    def __init__(self, learning_rate: float,  channels: int, classes: int, loss_func: Module, amp: bool = False, bilinear: bool = False):
        super().__init__()
        self.model = UNet(n_channels=channels, n_classes=classes, bilinear=bilinear)
        self.learning_rate = learning_rate
        self.amp = amp
        self.criterion = loss_func

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, true_masks = batch
        masks_pred_logits = self(images)
        # BCEWithLogitsLoss + dice_loss???
        loss = self.criterion(masks_pred_logits, true_masks.unsqueeze(1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, true_masks = batch
        masks_pred_logits = self(images)
        loss = self.criterion(masks_pred_logits, true_masks.unsqueeze(1))
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = RMSprop(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

class BiosensorDataModule(LightningDataModule):
    def __init__(self, data_path: str, batch_size: int, transform=None, biosensor_length=16, mask_size=80, mask_type=bool):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.transform = transform
        self.biosensor_length = biosensor_length
        self.mask_size = mask_size
        self.mask_type = mask_type

    def setup(self, stage=None):
        torch.manual_seed(42)
        # dataset
        dataset = BiosensorDataset(self.data_path, mask_type=self.mask_type, biosensor_length=self.biosensor_length, mask_size=self.mask_size)

        # split dataset
        train_size = int(len(dataset)*0.86)
        val_size = len(dataset) - train_size
        self.train_data, self.val_data = random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=True)


def main():
    args = get_args()
    data_dir = Path('data_with_centers/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNetLightningModule(learning_rate=args.lr, amp=args.amp, bilinear=args.bilinear)
    data_module = BiosensorDataModule(data_dir=data_dir, batch_size=args.batch_size)

    trainer = pl.Trainer(max_epochs=args.epochs, gpus=1 if torch.cuda.is_available() else 0, precision=16 if args.amp else 32)
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()