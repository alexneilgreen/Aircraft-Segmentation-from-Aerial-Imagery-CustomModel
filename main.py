import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.object_detection_model import ObjectDetectionModel
from PIL import Image
import numpy as np
import torch.nn as nn  # Import nn for loss functions

class CustomDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = os.listdir(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        label_path = os.path.join(self.labels_dir, self.image_files[idx].replace('.jpg', '.txt'))

        image = Image.open(img_path).convert("RGB")
        with open(label_path, 'r') as f:
            labels = f.readlines()

        # Process labels to get a single target class
        if labels:  # Check if there are any labels
            # Assume the format is <class_id> <x_center> <y_center> <width> <height>
            class_id, _, _, _, _ = map(float, labels[0].strip().split())
            target = class_id  # Use only the first class ID
        else:
            target = 0  # Default or placeholder class ID if no labels are found

        # Convert target to tensor
        target_tensor = torch.tensor(int(target), dtype=torch.long)  # Make sure it's long type for CrossEntropyLoss

        if self.transform:
            image = self.transform(image)

        return image, target_tensor
    
def calculate_accuracy(outputs, targets):
    _, predictions = torch.max(outputs, 1)
    correct = (predictions == targets).sum().item()
    return correct / targets.size(0)


def main(num_epochs=10, batch_size=16):
    # Parameters
    num_classes = 5  # Set this to the number of classes in your dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    # Dataset and DataLoader
    train_dataset = CustomDataset(images_dir='./train/images/', labels_dir='./train/labels/', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = CustomDataset(images_dir='./valid/images/', labels_dir='./valid/labels/', transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss Function, Optimizer
    model = ObjectDetectionModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        print(f"Epoch {epoch + 1} of {num_epochs}")

        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            correct_train += (outputs.argmax(1) == targets).sum().item()
            total_train += targets.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        model.eval()
        total_valid_loss = 0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for images, targets in valid_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                total_valid_loss += loss.item()
                correct_valid += (outputs.argmax(1) == targets).sum().item()
                total_valid += targets.size(0)

        valid_loss = total_valid_loss / len(valid_loader)
        valid_accuracy = correct_valid / total_valid

        model.eval()
        total_valid_loss = 0
        correct_valid = 0
        total_valid = 0

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")

        # Log performance to output file
        with open('output.txt', 'a') as file:
            file.write(f'Epoch [{epoch + 1}/{num_epochs}]\n'
                    f'Train Loss: {train_loss:.4f}\tTrain Accuracy: {train_accuracy:.4f}\n, '
                    f'Valid Loss: {valid_loss:.4f}\tValid Accuracy: {valid_accuracy:.4f}\n\n')

if __name__ == "__main__":
    # Set parameters here
    num_epochs = 30  # Change to 30
    batch_size = 16   # Change to 64
    main(num_epochs=num_epochs, batch_size=batch_size)
