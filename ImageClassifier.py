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
print(f"Image shape: {image.shape}")  # [1, 28, 28] -> 1 channel, 28x28 pixels
print(f"Label: {label}")  # integer 0-9

# Wrap in DataLoaders — this handles batching and shuffling for us
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Peek at one batch
images, labels = next(iter(train_loader))
print(f"Batch of images shape: {images.shape}")  # [64, 1, 28, 28]
print(f"Batch of labels shape: {labels.shape}")  # [64]

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
images, labels = next(iter(train_loader))  # adjust name if yours differs

print("Input batch shape:", images.shape)   # should be [batch_size, 1, 28, 28]

# Pass it through the model
outputs = model(images)
print("Output shape:", outputs.shape)       # should be [batch_size, 10]


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