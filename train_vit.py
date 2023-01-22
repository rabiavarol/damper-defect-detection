import dataset 
import torch
import models.vit_model 
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

if __name__ == "__main__":
    data_dir = "../dataset"
    train_ds, val_ds, test_ds = dataset.get_transformed_datasets(data_dir)
    id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label:id for id,label in id2label.items()}

    train_batch_size = 32
    eval_batch_size = 32

    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size, num_workers=20)
    val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=20)
    test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size)

    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=4,
        strict=True,
        verbose=True,
        mode='min'
    )

    vit_model = models.vit_model.ViTLightningModule(
        id2label=id2label, 
        label2id=label2id,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )
    logger = TensorBoardLogger("tb_logs", name="my_model")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(gpus=1, callbacks=[early_stop_callback, lr_monitor], log_every_n_steps=10, logger=logger)
    trainer.fit(vit_model)
    trainer.test()
    trainer.save_checkpoint("model_vit.ckpt")
