import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json
import os

# Define transforms for MNIST
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing for compatibility with ResNet input
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalization for grayscale images
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
}

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./Data/', train=True, transform=data_transforms['train'], download=True)
val_dataset = datasets.MNIST(root='./Data/', train=False, transform=data_transforms['val'], download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Define the models to train
models_to_train = {
    'resnet18': models.resnet18(pretrained=False),
    'resnet34': models.resnet34(pretrained=False),
    'resnet50': models.resnet50(pretrained=False),
    'pretrained_resnet18': models.resnet18(weights='ResNet18_Weights.DEFAULT'),
    'pretrained_resnet34': models.resnet34(weights='ResNet34_Weights.DEFAULT'),
    'pretrained_resnet50': models.resnet50(weights='ResNet50_Weights.DEFAULT')
}

os.makedirs("output/models/", exist_ok=True)
os.makedirs("output/logs/", exist_ok=True)

# Modify model function
def modify_model(model, num_classes):
    # Modify the final fully connected layer to match MNIST's 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Training function
def train_model(model, criterion, optimizer, num_epochs=25):
    best_acc = 0.0
    training_results = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Validate the model
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)

        training_results.append({
            'epoch': epoch,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc.item(),
            'val_loss': val_epoch_loss,
            'val_acc': val_epoch_acc.item()
        })

        print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

        # Save the best model
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), f'output/models/best_{model_name}.pth')

    return training_results

# Training loop for each model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model_name, model in models_to_train.items():
    # Modify the model for MNIST
    model = modify_model(model, num_classes=10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f'Training {model_name}...')
    training_results = train_model(model, criterion, optimizer, num_epochs=20)

    with open(f'output/logs/{model_name}_training_results.json', 'w') as f:
        json.dump(training_results, f)

    print(f'Finished training {model_name}')

print('All models trained and results saved.')
