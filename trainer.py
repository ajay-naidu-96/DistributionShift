import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

class CustomDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = CustomDataset('./Data/train/', transform=data_transforms['train'])
val_dataset = CustomDataset('./Data/test/', transform=data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Define the models
# models_to_train = {
#     'resnet18': models.resnet18(pretrained=False),
#     'resnet50': models.resnet50(pretrained=False),
#     'efficientnet_b0': models.efficientnet_b0(pretrained=False),
#     'pretrained_resnet18': models.resnet18(weights='ResNet18_Weights.DEFAULT'),
#     'pretrained_resnet50': models.resnet50(weights='ResNet50_Weights.DEFAULT'),
#     'efficientnet_b0': models.efficientnet_b0(pretrained=False)
# }

model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

num_classes = 150 
model.fc = nn.Linear(model.fc.in_features, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {i}/{len(train_loader)}, Loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

    model.train()