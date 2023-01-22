import torch
from datasets import load_dataset, DatasetDict
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

def split_dataset(data_dir):
    dataset = load_dataset("imagefolder", data_dir=data_dir)
    dataset = dataset['train']
    splits = dataset.train_test_split(test_size=0.2)
    train_ds = splits['train']
    test_ds = splits['test']
    train_valid_dataset = DatasetDict({'train': train_ds})
    splits = train_valid_dataset['train'].train_test_split(test_size=0.2)
    train_ds = splits['train']
    val_ds = splits['test']

    return train_ds, val_ds, test_ds 

def train_transforms(examples):
    normalize = Normalize(mean=[x/255.0 for x in [0.485, 0.456, 0.406]],
                            std=[x/255.0 for x in [0.229, 0.224, 0.225]])
    train_transforms = Compose(
            [
                Resize((224,224)),
                ToTensor(),
                normalize,
            ]
        )
    examples['pixel_values'] = [train_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def val_transforms(examples):
    normalize = Normalize(mean=[x/255.0 for x in [0.485, 0.456, 0.406]],
                            std=[x/255.0 for x in [0.229, 0.224, 0.225]])
    val_transforms = Compose(
            [
                Resize((224,224)),
                ToTensor(),
                normalize,
            ]
        )
    examples['pixel_values'] = [val_transforms(image.convert("RGB")) for image in examples['image']]
    return examples

def set_transforms(train_ds, val_ds, test_ds):
    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)
    test_ds.set_transform(val_transforms)
    return train_ds, val_ds, test_ds

def get_transformed_datasets(data_dir):
    train_ds, val_ds, test_ds = split_dataset(data_dir=data_dir)
    return set_transforms(train_ds, val_ds, test_ds)