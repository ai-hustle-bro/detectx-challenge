import numpy as np
import torch
import torchvision.transforms as transforms

def get_accuracy(outputs, labels):
    """
    Calculates the accuracy between model outputs and true labels.
    """
    _, preds = torch.max(outputs, 1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def get_transforms(img_size=224):
    """
    Standard preprocessing transforms for computer vision tasks.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def encode_labels(labels_list):
    """
    Encodes string labels into integers if needed.
    (This is optional depending on how the CSV is structured).
    """
    unique_labels = sorted(list(set(labels_list)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for i, label in enumerate(unique_labels)}
    return label_to_id, id_to_label
