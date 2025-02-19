import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Define a simple CNN model for facial expression recognition
class ExpressionCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ExpressionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Automatically determine the flattened size
        self.flatten_size = 32 * 56 * 56  # Adjust based on input size

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)  # Output layer for 5 classes

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten dynamically
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Custom Dataset to Ensure Labels Stay in the Range of 0-4
class CustomDataset(Dataset):
    def __init__(self, original_dataset, num_classes=5):
        self.dataset = original_dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label = label % self.num_classes  # Ensure labels are within 0-4
        return image, label

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset (replace with actual dataset if available)
original_dataset = torchvision.datasets.FakeData(transform=transform)

# Wrap dataset to ensure label correction
train_dataset = CustomDataset(original_dataset, num_classes=5)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, optimizer
model = ExpressionCNN(num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
num_epochs = 5
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)

        # Ensure labels are long type and within range
        labels = labels.long()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "model.pth")
print("Model saved as 'model.pth'")
