# LOAD DATASET
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Convert images to tensors (0-255 pixels -> 0-1 range)
transform = transforms.ToTensor()

# Download and load the training and test sets
train_data = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_data = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# Look at a single sample
image, label = train_data[0]
print(f"Image shape: {image.shape}") 
print(f"Label: {label}")  

# Wrap in DataLoaders — this handles batching and shuffling for us
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Peek at one batch
images, labels = next(iter(train_loader))
print(f"Batch of images shape: {images.shape}")  
print(f"Batch of labels shape: {labels.shape}")  

# DEFINE THE MODEL
# import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()          # 28x28 image -> 784 vector
        self.fc1 = nn.Linear(784, 128)        # input layer -> hidden layer
        self.fc2 = nn.Linear(128, 64)         # hidden -> hidden
        self.fc3 = nn.Linear(64, 10)          # hidden -> output (10 classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))               # activation function
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                       # raw logits, no activation here
        return x

model = SimpleNN()
print(model)

# Checking work uptill now
# Grab one batch from your dataloader
images, labels = next(iter(train_loader)) 
print("Input batch shape:", images.shape)   

# Pass it through the model
outputs = model(images)
print("Output shape:", outputs.shape)      


# LOSS FUNCTION AND OPTIMIZER
# Loss function - measures how wrong the predictions are
criterion = nn.CrossEntropyLoss()

# Optimizer - updates weights using gradients
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# TRAINING LOOP
epochs = 5

for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in train_loader:
        # 1. Forward pass - get predictions
        outputs = model(images)

        # 2. Calculate loss - how wrong were we?
        loss = criterion(outputs, labels)

        # 3. Backward pass - compute gradients
        optimizer.zero_grad()   # clear old gradients
        loss.backward()         # backpropagation

        # 4. Update weights - gradient descent step
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


# EVALUATION
# Set model to evaluation mode
model.eval()

correct = 0
total = 0

# No gradients needed for evaluation - saves memory and computation
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # get the class with highest score

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Training weights - Saving and Loading the Model
# Save the trained model's weights
torch.save(model.state_dict(), 'fashion_mnist_model.pth')
print("Model saved successfully!")
# Recreate the model architecture first
model = SimpleNN()

# Load the saved weights into it
model.load_state_dict(torch.load('fashion_mnist_model.pth'))
model.eval()  # set to evaluation mode
print("Model loaded successfully!")

import matplotlib.pyplot as plt

# Class names for Fashion-MNIST (in label order 0-9)
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Grab a single image from the test set
image, true_label = test_data[100] 

model.eval()
with torch.no_grad():
    output = model(image.unsqueeze(0))  
    _, predicted = torch.max(output, 1)

# Show the image with prediction
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Predicted: {classes[predicted.item()]} | Actual: {classes[true_label]}")
plt.axis('off')
plt.show()

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers - detect spatial patterns
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers - final classification
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = x.view(x.size(0), -1)               # flatten for fc layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

cnn_model = CNN()
print(cnn_model)

# ===== Train the CNN =====

# Loss + optimizer for the CNN (separate from SimpleNN's)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):
    running_loss = 0.0
    cnn_model.train()   # set to training mode (matters because of Dropout)

    for images, labels in train_loader:
        outputs = cnn_model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"CNN Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# ===== Evaluate the CNN =====
cnn_model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

cnn_accuracy = 100 * correct / total
print(f"CNN Test Accuracy: {cnn_accuracy:.2f}%")

# ===== Save the CNN =====
torch.save(cnn_model.state_dict(), 'cnn_model.pth')
print("CNN model saved successfully!")