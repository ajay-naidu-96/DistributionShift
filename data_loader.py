import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
from torch.utils.data import DataLoader
from torchvision import transforms


def train_val_dataset(dataset, labels, val_split=0.25):

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, 
                                        stratify=labels, random_state=42)

    datasets = {}

    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)

    return datasets


def get_splits(path):

    transform_train = transforms.Compose([
    transforms.Resize((224, 224)),

    # transforms.RandomApply([transforms.RandomCrop(200)], p=0.5),    
    transforms.RandomHorizontalFlip(p=0.5),                        
    transforms.RandomRotation(degrees=15),                         
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                            saturation=0.2, hue=0.1),              
    transforms.ToTensor(),                                     
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])              
    ])

    dataset = ImageFolder(path, transform=transform_train)  
      
    labels = [target for _, target in dataset]

    datasets = train_val_dataset(dataset, labels)

    return datasets