import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import resnet50, ResNet50_Weights


class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self, train_dataloader, val_dataloader, test_dataloader):
        super().__init__()

        self.save_hyperparameters()
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader

        self.num_classes = 2
        self.model_ft = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.num_ftrs = self.model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        self.model_ft.fc = nn.Sequential(
            nn.Linear(self.num_ftrs, int(self.num_ftrs/4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.num_ftrs/4), int(self.num_ftrs/16)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.num_ftrs/16), int(self.num_ftrs/64)),
            nn.ReLU(inplace=True),
            nn.Linear(int(self.num_ftrs/64), 2)
        )
        
    def forward(self, x):
        x = self.model_ft(x)
        return x

    def common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]

        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)
        self.log_dict({'test_loss': loss, 'test_accuracy': accuracy})  
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        lr_schedulers = {"scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6,
        verbose=True), "monitor": "validation_loss"}
        return [optimizer], [lr_schedulers]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
