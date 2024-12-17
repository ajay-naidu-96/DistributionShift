from common_imports import torch, nn, torchvision, transforms, optim, os
import resnet
from trainer_v2 import train_model
import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from data_loader import get_splits
from model_bed import get_models, get_simple_cnn

os.makedirs("output/models/", exist_ok=True)
os.makedirs("output/logs/", exist_ok=True)

if __name__ == "__main__":

    num_classes = 10
    num_epochs = 10

    dataset_paths = {'fixed_color':'./Data/cmnist/fixed/train/', 
                    'rand_color':'./Data/cmnist/random/train/'}

    models_dict = get_models(num_classes)

    # models_dict = get_simple_cnn(num_classes)

    for dataset_type, dataset_path in  dataset_paths.items():

        dataset = get_splits(dataset_path)

        train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=128, num_workers=4, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset['val'], batch_size=128, num_workers=4, shuffle=False)

        for model_name, model in models_dict.items():
            
            model_name = model_name + "_" + dataset_type

            train_model(model, train_loader, test_loader, model_name, num_epochs)









