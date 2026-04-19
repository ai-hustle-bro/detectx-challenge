import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from utils import get_transforms

class CompetitionDataset(Dataset):
    def __init__(self, data_dir, csv_file=None, transform=None, is_test=False):
        """
        Args:
            data_dir (string): Directory with all the images.
            csv_file (string, optional): Path to the labels CSV file (for train).
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Whether the dataset is for testing (no labels).
        """
        self.data_dir = data_dir
        self.transform = transform if transform else get_transforms()
        self.is_test = is_test

        if not self.is_test:
            self.labels_df = pd.read_csv(csv_file)
            self.image_ids = self.labels_df['image_id'].values
            self.labels = self.labels_df['label'].values
        else:
            # For test set, we just list all files in the directory
            self.image_ids = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        
        # Ensure img_id has extension if it's from CSV
        if not self.is_test and not str(img_id).endswith(('.jpg', '.png', '.jpeg')):
             img_path = os.path.join(self.data_dir, f"{img_id}.jpg")
        else:
             img_path = os.path.join(self.data_dir, img_id)

        # Load image using OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Convert BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            # Return image and its ID for submission tracking
            return image, img_id
        else:
            # Return image and its label
            label = self.labels[idx]
            return image, label
