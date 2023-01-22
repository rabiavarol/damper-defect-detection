import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import ViTForImageClassification, AdamW

class ViTLightningModule(pl.LightningModule):
    def __init__(self, id2label, label2id, train_dataloader, val_dataloader, test_dataloader, num_labels=2):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                              num_labels=2,
                                                              id2label=id2label,
                                                              label2id=label2id,
                                                              output_attentions=True)
        self.save_hyperparameters()
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits
        
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
        optimizer = AdamW(self.parameters(), lr=5e-6)
        lr_schedulers = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
        factor=0.1, patience=2, threshold=0.0001, threshold_mode='abs'), "monitor": "validation_loss"}
        return [optimizer], [lr_schedulers]


    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
