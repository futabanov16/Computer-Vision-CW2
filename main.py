import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from CNN_dataset import CNN_Dataset
from model import AlexNet
import numpy as np

def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.01
    num_epochs = 100
    weight_decay = 5e-4
    momentum = 0.9

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    dataset = CNN_Dataset(data_dir='training', train=True)
    # Split dataset into training and validation sets
    # Use 90% for training and 10% for validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    

    # Initialize model, loss function, and optimizer
    model = AlexNet(num_classes=15) 
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    model.to(device)
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for idx, data in enumerate(train_loader):
            # Forward pass
            images = data[0].to(device).float()
            labels = data[1].to(device).long()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # Validation loop
        with torch.inference_mode():
            model.eval()
            correct = 0
            total = 0
            for idx, data in enumerate(val_loader):
                images = data[0].to(device).float()
                labels = data[1].to(device).long()
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Validation Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    main()
