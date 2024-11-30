from common_imports import torch, nn, torchvision, transforms, optim, json

def train_model(model, train_loader, test_loader, model_name, num_epochs=25):

    print(f'Training {model_name}...')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    training_results = []

    model.to(device)

    for epoch in range(num_epochs):

        print("Epoch: {}".format(epoch))

        model.train()

        running_loss = 0.0
        running_corrects = 0

        for i, (inputs, labels) in enumerate(train_loader):
            
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

        model.eval()

        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_loader:

                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(predicted == labels.data)

        val_epoch_loss = val_running_loss / len(test_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(test_loader.dataset)

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

    
    with open(f'output/logs/{model_name}_training_results.json', 'w') as f:
        json.dump(training_results, f)

    print(f'Finished training {model_name}')