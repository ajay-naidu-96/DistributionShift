import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def get_models(num_classes, pre_train=True):
    models_dict = {}

    # ResNet-18
    model_resnet18 = models.resnet18(pretrained=pre_train)
    model_resnet18.fc = nn.Linear(model_resnet18.fc.in_features, num_classes)
    models_dict["resnet18"] = model_resnet18

    # EfficientNet-B0
    model_efficientnet_b0 = models.efficientnet_b0(pretrained=pre_train)
    model_efficientnet_b0.classifier[1] = nn.Linear(model_efficientnet_b0.classifier[1].in_features, num_classes)
    models_dict["efficientnet_b0"] = model_efficientnet_b0

    # VGG-16
    model_vgg16 = models.vgg16(pretrained=pre_train)
    model_vgg16.classifier[6] = nn.Linear(model_vgg16.classifier[6].in_features, num_classes)
    models_dict["vgg16"] = model_vgg16

    return models_dict

def get_simple_cnn(num_classes):

    models_dict = {}

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            
            # Convolutional Layer 1
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
            
            # Convolutional Layer 2
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            
            # Convolutional Layer 3
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
            
            # MaxPooling Layer
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            
            # Fully Connected Layer 1 (fc1) and Fully Connected Layer 2 (fc2)
            self.fc1 = nn.Linear(128 * 3 * 3, 128)  # Adjusted based on new feature map size
            self.fc2 = nn.Linear(128, 10)  # 10 classes for classification (Fashion-MNIST)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))  # Apply Conv1 + ReLU
            x = self.pool(x)  # Max pooling
            
            x = F.relu(self.conv2(x))  # Apply Conv2 + ReLU
            x = self.pool(x)  # Max pooling
            
            x = F.relu(self.conv3(x))  # Apply Conv3 + ReLU
            x = self.pool(x)  # Max pooling
            
            # Flatten the tensor before feeding into fully connected layers
            # print("Shape before flattening:", x.shape)  # Debugging line
            
            x = x.view(-1, 128 * 3 * 3)  # Flattening to match the actual size for fc1
            
            x = F.relu(self.fc1(x))  # Apply FC1 + ReLU
            x = self.fc2(x)  # Output layer
            return x

    models_dict["simple_cnn"] = SimpleCNN()

    return models_dict


def get_older_models(num_classes):
    
    print("You Don't Want to Be Here")

    # Define transformations for the training and test sets
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.1307,), (0.3081,))
    # ])

    # train_dataset = torchvision.datasets.MNIST(root='./Data', train=True, download=True, transform=transform)
    # test_dataset = torchvision.datasets.MNIST(root='./Data', train=False, download=True, transform=transform)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # model_name = 'resnet_18_mnist'
    # model_18 = resnet.ResNet18()
    # results = train_model(model_18, train_loader, test_loader, model_name)

    # model_name = 'resnet_34_mnist'
    # model_34 = resnet.ResNet34()
    # train_model(model_34, train_loader, test_loader, model_name)

    # model_name = 'resnet_50_mnist'
    # model_50 = resnet.ResNet50()
    # train_model(model_50, train_loader, test_loader, model_name)

