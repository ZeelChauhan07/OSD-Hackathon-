# Image-Classifier

# Fashion-MNIST Image Classifier

A PyTorch-based image classifier that identifies clothing items from the Fashion-MNIST dataset, built and compared across two architectures — a simple feedforward network and a convolutional neural network (CNN). Built as a hands-on learning project during a 5-day ML hackathon, focused on understanding core deep learning concepts by implementing and comparing model architectures from scratch.

## Description
This project walks through the essential stages of a supervised deep learning workflow: loading and preparing image data, designing neural networks, training with backpropagation and gradient descent, and evaluating generalization on unseen data. It goes a step further by comparing a basic feedforward network against a CNN, demonstrating in practice why spatial-aware architectures outperform simpler ones on image data.

## Features
- Loads and preprocesses the Fashion-MNIST dataset using `torchvision`
- Two model architectures for direct comparison:
  - `SimpleNN` — a 3-layer feedforward network
  - `CNN` — a convolutional network with pooling and dropout
- Full training loop with forward pass, loss computation, and backpropagation
- Model evaluation on a held-out test set to measure real-world accuracy
- Model save/load functionality for both architectures
- Single-image inference demo with visual output
- Clear, step-by-step, checkpoint-driven development

## Technologies Used
- **Python 3.12**
- **PyTorch** — model building, training, autograd
- **torchvision** — dataset loading and transforms
- **matplotlib** — visualizing predictions
- **Fashion-MNIST** — 70,000 grayscale clothing images across 10 classes
- **VSCode** (Windows) — development environment

## Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# Create and activate a virtual environment (Python 3.12)
py -3.12 -m venv venv
venv\Scripts\activate      # Windows

# Install dependencies
pip install torch torchvision matplotlib

## Usage
# Run the main script (loads data, trains both models, evaluates, runs inference demo)
python ImageClassifier.py

## Project Structure
Model Architecture
├── ImageClassifier.py       # Main script: data loading, models, training, evaluation, inference
├── fashion_mnist_model.pth  # Saved SimpleNN weights
├── cnn_model.pth             # Saved CNN weights
├── README.md                  # Project documentation
└── data/                       # Fashion-MNIST dataset (auto-downloaded on first run)

## Model Architectures
# SimpleNN(feedforward)
SimpleNN(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc1): Linear(in_features=784, out_features=128)
  (fc2): Linear(in_features=128, out_features=64)
  (fc3): Linear(in_features=64, out_features=10)
)

# CNN(convolutional)
CNN(
  (conv1): Conv2d(1, 32, kernel_size=3, padding=1)
  (conv2): Conv2d(32, 64, kernel_size=3, padding=1)
  (pool): MaxPool2d(kernel_size=2, stride=2)
  (fc1): Linear(in_features=3136, out_features=128)
  (fc2): Linear(in_features=128, out_features=10)
  (dropout): Dropout(p=0.25)
)

## MODEL - TEST ACCURACY
SimpleNN - 86.67%
CNN - 91.13%










# # Fashion-MNIST Image Classifier

# A simple feedforward neural network built with PyTorch to classify clothing items through image classification from the Fashion-MNIST dataset. This project was built as part of a 5-day ML hackathon to learn core deep learning concepts through hands-on implementation.

# # Project Goal
# Understand and implement the full ML pipeline — data loading, model design, training, and evaluation using a real dataset and framework, not just theory.

# # Tech Stack
# - Python 3.12
# - PyTorch + torchvision
# - Fashion-MNIST dataset

# # Progress

# - Loaded Fashion-MNIST dataset via `torchvision.datasets` and `DataLoader`
# - Defined `SimpleNN`: a 3-layer feedforward neural network
#   - Input: 784 (28x28 flattened image)
#   - Hidden layers: 128 → 64
#   - Output: 10 classes
# - Implemented training loop (forward pass, loss calculation, backpropagation, optimizer step)
# - Trained for 5 epochs — loss decreased from **0.5676 → 0.3099**
# - Evaluated on unseen test set — **87.40% accuracy**

# # Usage
# - Run the training script (loads data, builds model, trains, evaluates)
#   python ImageClassifier.py

# # Project Structure

# ├── ImageClassifier.py            # Main script: data loading, modeltraining, evaluation
# ├── README.md            # Project documentation
# └── data/                 # Fashion-MNIST dataset (auto-downloaded on first run)

# # Model Architecture
# SimpleNN(
# (flatten): Flatten(start_dim=1, end_dim=-1)
# (fc1): Linear(in_features=784, out_features=128)
# (fc2): Linear(in_features=128, out_features=64)
# (fc3): Linear(in_features=64, out_features=10)
# )

# # Next Steps
# - Save and load trained model weights
# - Single-image inference demo
# - (Optional) Explore improvements: more epochs, CNN architecture, hyperparameter tuning

# # How to Run
# ```bash
# python -m venv venv
# venv\Scripts\activate  # Windows
# pip install torch torchvision
# python ImageClassifier.py