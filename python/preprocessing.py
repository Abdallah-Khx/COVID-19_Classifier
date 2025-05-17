import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import mlflow
from sklearn.model_selection import train_test_split

class COVIDDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class DataPreprocessor:
    def __init__(self, img_size=299):
        self.img_size = img_size
        self.classes = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_all_images_and_labels(self, root_dir):
        image_paths = []
        labels = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name, "images")
            if os.path.exists(class_dir):
                class_images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
                for img_name in class_images:
                    image_paths.append(os.path.join(class_dir, img_name))
                    labels.append(self.class_to_idx[class_name])
        return np.array(image_paths), np.array(labels)

    def split_and_save(self, root_dir, split_dir="splits", seed=42):
        os.makedirs(split_dir, exist_ok=True)
        image_paths, labels = self.get_all_images_and_labels(root_dir)
        
        # Debug: Print total samples per class
        print("\nTotal samples per class:")
        for class_idx, class_name in enumerate(self.classes):
            count = np.sum(labels == class_idx)
            print(f"{class_name}: {count} samples")
        
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_paths, labels, 
            test_size=0.3, 
            stratify=labels,  # This ensures class distribution is preserved
            random_state=seed
        )
        
        # Second split: 50% validation, 50% test from the temp set
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=0.5, 
            stratify=y_temp,  # This ensures class distribution is preserved
            random_state=seed
        )
        
        # Debug: Print split sizes
        print("\nSplit sizes:")
        print(f"Train: {len(X_train)} samples")
        print(f"Validation: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")
        
        # Debug: Print class distribution in each split
        for split_name, (X, y) in [("Train", (X_train, y_train)), 
                                  ("Validation", (X_val, y_val)), 
                                  ("Test", (X_test, y_test))]:
            print(f"\n{split_name} set class distribution:")
            for class_idx, class_name in enumerate(self.classes):
                count = np.sum(y == class_idx)
                print(f"{class_name}: {count} samples")
        
        # Save splits
        np.save(os.path.join(split_dir, "train_images.npy"), X_train)
        np.save(os.path.join(split_dir, "train_labels.npy"), y_train)
        np.save(os.path.join(split_dir, "val_images.npy"), X_val)
        np.save(os.path.join(split_dir, "val_labels.npy"), y_val)
        np.save(os.path.join(split_dir, "test_images.npy"), X_test)
        np.save(os.path.join(split_dir, "test_labels.npy"), y_test)
        print(f"\nSaved splits to {split_dir}")

    def load_split(self, split_dir, split):
        images = np.load(os.path.join(split_dir, f"{split}_images.npy"))
        labels = np.load(os.path.join(split_dir, f"{split}_labels.npy"))
        return images, labels

    def create_data_loaders_from_split(self, split_dir, batch_size=32, num_workers=4):
        train_images, train_labels = self.load_split(split_dir, "train")
        val_images, val_labels = self.load_split(split_dir, "val")
        test_images, test_labels = self.load_split(split_dir, "test")
        train_dataset = COVIDDataset(train_images, train_labels, transform=self.train_transform)
        val_dataset = COVIDDataset(val_images, val_labels, transform=self.val_transform)
        test_dataset = COVIDDataset(test_images, test_labels, transform=self.val_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader, test_loader

    def log_preprocessing_params(self):
        with mlflow.start_run(nested=True):
            mlflow.log_params({
                "image_size": self.img_size,
                "batch_size": 32,
                "augmentation_enabled": True
            }) 