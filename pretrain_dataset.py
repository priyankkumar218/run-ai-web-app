import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import pandas as pd
import clip

class ImageTextDataset(Dataset):
    def __init__(self, images_folder, annotations_path, tokenize, max_captions=4, transform=None):
        self.images_folder = images_folder
        self.annotations_df = self.load_annotations(annotations_path, max_captions)
        self.transform = transform
        self.tokenize = tokenize

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        img_name, captions = self.annotations_df.iloc[idx]
        img_path = os.path.join(self.images_folder, f"{img_name:012d}.jpg")
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        
        annotation = self.tokenize(captions[0])

        return image, annotation

    def load_annotations(self, annotations_path, max_captions):
        with open(annotations_path, 'r') as f:
            annotations_data = json.load(f)

        # Flatten the nested structure and select relevant columns
        annotations_df = pd.json_normalize(annotations_data['annotations'], sep='_')[['image_id', 'caption']]

        # Combine multiple captions for the same image
        annotations_df = annotations_df.groupby('image_id')['caption'].apply(list).reset_index(name='captions')

        # Limit the number of captions per image
        annotations_df['captions'] = annotations_df['captions'].apply(lambda x: x[:max_captions])

        return annotations_df

def get_dataloaders(batch_size=32, num_workers=4):
    IMG_DIM = 224

    # Custom collate function to handle non-tensor data types
    def custom_collate(batch):
        images, captions = zip(*batch)
        images = torch.stack([transforms.ToTensor()(img) for img in images])
        return images, captions

    # Resize and crop the images to 256x256
    image_transform = transforms.Compose([
        transforms.Resize((IMG_DIM, IMG_DIM)),
        transforms.CenterCrop((IMG_DIM, IMG_DIM)),
    ])

    # Example usage:
    # Specify the paths to your image folders and annotation JSON file
    train_images_path = "train2017"
    val_images_path = "val2017"
    captions_train_path = "annotations/captions_train2017.json"
    captions_val_path = "annotations/captions_val2017.json" 

    # Instantiate the dataset with the CLIP tokenizer and image transform
    train_dataset = ImageTextDataset(images_folder=train_images_path, annotations_path=captions_train_path, max_captions=4, transform=image_transform, tokenize=clip.tokenize)
    val_dataset = ImageTextDataset(images_folder=val_images_path, annotations_path=captions_val_path, max_captions=4, transform=image_transform, tokenize=clip.tokenize)

    # Example usage of DataLoader with custom collate function
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, num_workers=num_workers)

    return train_dataloader, val_dataloader