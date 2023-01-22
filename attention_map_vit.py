import numpy as np
import cv2
import matplotlib.pyplot as plt
import dataset 
import torch
import models.vit_model 
from torch.utils.data import DataLoader
from PIL import Image
from einops import reduce, rearrange


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

if __name__ == "__main__":
    # directory of the dataset
    data_dir = "../dataset"
    train_ds, val_ds, test_ds = dataset.get_transformed_datasets(data_dir)
    id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label:id for id,label in id2label.items()}

    train_batch_size = 1
    eval_batch_size = 1

    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size, num_workers=8)
    val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=8)
    test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=8)

    device = torch.device('cpu')

    model = models.vit_model.ViTLightningModule(
        id2label=id2label, 
        label2id=label2id,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )
    model = models.vit_model.ViTLightningModule.load_from_checkpoint(checkpoint_path="all_colored_z_vit_v16.ckpt")

    intact_num = 0
    defect_mask_total=torch.zeros(224, 224)
    defect_num = 0
    intact_mask_total=torch.zeros(224, 224)

    for i in range(test_ds.shape[0]):
        image = test_ds[i]
        x = image["pixel_values"]
        x = torch.unsqueeze(x, 0)

        outputs = model.vit(x)
        logits = outputs.logits
        attentions = outputs.attentions
        
        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        att_mat = torch.stack(attentions).squeeze(1)

        # attention
        att_mat = reduce(att_mat, 'b h len1 len2 -> b len1 len2', 'mean')
        im = np.array(cv2.resize(np.array(image["image"]), (224,224), interpolation = cv2.INTER_AREA)).transpose(2,0,1)

        residual_att = torch.eye(att_mat.size(1))
        aug_att_mat = att_mat + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
            
        # Attention from the output token to the input space.
        v = joint_attentions[-1]
        grid_size = int(np.sqrt(aug_att_mat.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        mask = cv2.resize(mask / mask.max(), (im.shape[1], im.shape[2]))[np.newaxis, np.newaxis, ...]

        result = (mask* im).squeeze(axis=0).transpose(1,2,0)/255)

        mask = mask.squeeze()

        if predicted_class_idx == 0:
            defect_num += 1
            defect_mask_total += mask
        else:
            intact_num += 1
            intact_mask_total += mask

    intact_mask_total /= intact_num
    defect_mask_total /= defect_num



    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(intact_mask_total)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('tight')
    ax.axis('off')
    plt.savefig("intact_avg_vit.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(defect_mask_total)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('tight')
    ax.axis('off')
    plt.savefig("defect_avg_vit.png")
