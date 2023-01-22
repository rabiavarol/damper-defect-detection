import cv2
import dataset 
import torch
import models.cnn_model 
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from PIL import Image

# This class stores the feature maps of the last conv layer of the image, 
# after a forward pass.

class FeatureBuffer():

    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, _input, _output): 
        self.features = _output

    def remove(self): 
        self.hook.remove()
    
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

if __name__ == "__main__":
    # Directory for the dataset
    data_dir = "../dataset"
    train_ds, val_ds, test_ds = dataset.get_transformed_datasets(data_dir)
    id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
    label2id = {label:id for id,label in id2label.items()}

    train_batch_size = 1
    eval_batch_size = 1

    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=train_batch_size, num_workers=20)
    val_dataloader = DataLoader(val_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=20)
    test_dataloader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=eval_batch_size, num_workers=20)

    new_model = models.cnn_model.ImagenetTransferLearning(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )

    new_model = models.cnn_model.ImagenetTransferLearning.load_from_checkpoint(checkpoint_path="./all_colored_Z_cnn_expLR_extra_layers.ckpt")
    final_conv_layer = new_model.model_ft.layer4
    # Register hook.
    fb = FeatureBuffer(final_conv_layer)

    defect_num = 0
    defect_cam_total = torch.zeros(432,432)
    intact_num = 0
    intact_cam_total = torch.zeros(432,432)
    
    for i in range(test_ds.shape[0]):
        x = test_ds[i]['pixel_values']
        img = test_ds[i]['image']

        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.savefig('./tmp/' + str(i) + '.png')

        input_tensor = torch.unsqueeze(x, 0)

        # Model in evaluation mode.
        new_model = new_model.eval()

        # Get probabilities and feature maps.
        with torch.no_grad():
            out = new_model(input_tensor)

        feature_maps = fb.features
        predicted_label = out.argmax(-1).item()

        # 2 classes, and 2048xWxH feature maps ((W,H) depends on image's size after pre-processing).

        probs = torch.nn.functional.softmax(out[0], dim=0)
        score, _class = torch.max(probs, dim=0)
        # Get weights and biases for fully conected linear layer.
        weights_and_biases = list(new_model.model_ft.fc.parameters())

        # Get weights for the class's neuron.
        class_weights = weights_and_biases[0][_class]


        # Weights must be reshaped to match feature maps's dimension.
        class_weights = class_weights.reshape((-1, 1, 1))

        # We can also reduce one empty dimension (first one) of the feature maps.
        feature_maps = feature_maps.flatten(start_dim=0, end_dim=1)

        # Get class_activation maps
        class_activation_maps = np.array(torch.sum(feature_maps * class_weights, dim=0).detach(), dtype=np.float32)

        plt.figure(figsize=(6, 6))
        plt.imshow(class_activation_maps)
        plt.savefig('./tmp/' + str(i) + '_test.png')
        # Resize tensor to match original image's size.
        resized_cam = cv2.resize(class_activation_maps, dsize=(432,432), interpolation=cv2.INTER_LANCZOS4)
        plt.figure(figsize=(6, 6))
        plt.imshow(resized_cam)
        plt.savefig('./tmp/' + str(i) + '_cam.png')

        plt.figure(figsize=(6, 6))
        plt.imshow(img, alpha=1)
        plt.imshow(resized_cam, alpha=0.4)
        plt.savefig('./tmp/' + str(i) + '_merged.png')

        im1 = cv2.imread('./tmp/' + str(i) + '.png')
        im2 = cv2.imread('./tmp/' + str(i) + '_cam.png')
        im3 = cv2.imread('./tmp/' + str(i) + '_merged.png')
        im_h = cv2.hconcat([im1, im2, im3])
        
        if id2label[predicted_label] == 'defect':
            defect_num += 1
            defect_cam_total += resized_cam
            cv2.imwrite('./class_activation_maps/defect/' + str(i) + '.png', im_h)
        else:
            intact_num += 1
            intact_cam_total += resized_cam
            cv2.imwrite('./class_activation_maps/intact/' + str(i) + '.png', im_h)
    
    # Find average heat map image on test set
    intact_cam_total /= intact_num
    defect_cam_total /= defect_num

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(intact_cam_total)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('tight')
    ax.axis('off')
    plt.savefig("intact_avg_cam.png")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(defect_cam_total)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('tight')
    ax.axis('off')
    plt.savefig("defect_avg_cam.png")
