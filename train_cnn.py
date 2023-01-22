import dataset 
import torch
import models.cnn_model 
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

    train_batch_size = 64
    eval_batch_size = 64

    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size, num_workers=20)
    val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=20)
    test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size)
  
    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=4,
        strict=False,
        verbose=False,
        mode='min'
    )

    cnn_model = models.cnn_model.ImagenetTransferLearning(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )

    logger = TensorBoardLogger("tb_logs_cnn", name="my_model")
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(gpus=1, accelerator="gpu", callbacks=[early_stop_callback, lr_monitor], log_every_n_steps=10, logger=logger)
    trainer.fit(cnn_model)
    trainer.test()
    trainer.save_checkpoint("model_cnn.ckpt")
    
    # After training is done, you can use these lines for inference
    """
    new_model = models.cnn_model.ImagenetTransferLearning.load_from_checkpoint(checkpoint_path="./model_cnn.ckpt")
    correct_pred = 0
    false_pred = 0
    for i in range(test_ds.shape[0]):
        
        x = test_ds[i]['pixel_values']
        input_tensor = torch.unsqueeze(x, 0)
        # # Model in evaluation mode.
        new_model = new_model.eval()
    
        with torch.no_grad():
            out = new_model(input_tensor)

        predicted_label = out.argmax(-1).item()
        if (id2label[predicted_label] == id2label[test_ds[i]['label']]):
            correct_pred += 1
        else: 
            false_pred += 1

    print("correct_pred", correct_pred) 
    print("false_pred", false_pred)    
    """
    #print("Predicted   : ", id2label[predicted_label])
    #print("Ground Truth: ", id2label[test_ds[i]['label']])
