# PyTorch core libraries
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn tools for data and evaluation
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

LEARNING_RATE = 0.02
EPOCHS = 2000
BATCH_SIZE = 64

# GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch sees CUDA?:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print(f"Using device: {device}")

# Load and prepare data
def load_data(batch_size=64):
    # Load digits dataset (images: 8x8 pixels → 64 features)
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Normalize input features to zero mean, unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Train-test split (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Move data to GPU if available
    if torch.cuda.is_available():
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        y_train = y_train.to(device)
        y_test = y_test.to(device)

    # Wrap training data in DataLoader for mini-batch SGD
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_data = (X_test, y_test)

    return train_loader, test_data

# Define a single-layer neural network
class SingleLayerNet:
    def __init__(self, input_dim=64, output_dim=10):
        torch.manual_seed(42)
        # Initialize weights and bias with gradient tracking
        self.W = torch.randn(input_dim, output_dim, requires_grad=True, device=device)
        self.b = torch.zeros(output_dim, requires_grad=True, device=device)
        # Store model parameters in a list
        self.params = [self.W, self.b]

    def forward(self, x):
        # Compute output: linear transformation XW + b
        return x @ self.W + self.b

# Train the model using SGD
def train_model(model, train_loader, lr=0.01, epochs=1000):
    optimizer = torch.optim.SGD(model.params, lr=lr)
    losses = []

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            # Move each batch to GPU if available
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            logits = model.forward(X_batch)
            loss = F.cross_entropy(logits, y_batch)

            # Backward pass and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses.append(loss.item())
        # Print loss every 500 epochs
        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

    return losses

def plot_loss(losses: list):
    """
    Plot the training loss over epochs.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Training")
    plt.legend()
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot the confusion matrix using a heatmap."
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(cm), annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.show()

# Evaluate model performance on test data
def evaluate_model(model, test_data, losses):
    X_test, y_test = test_data

    with torch.no_grad():
        # Forward pass on test data
        logits = model.forward(X_test)
        predictions = torch.argmax(logits, dim=1)

        # Compute accuracy
        accuracy = (predictions == y_test).float().mean()
        print(f"\nTest Accuracy: {accuracy.item():.4f}")

        # Print classification report
        print("\nClassification Report:\n", classification_report(y_test.cpu(), predictions.cpu()))

        # Plot confusion matrix
        plot_confusion_matrix(y_test.cpu(), predictions.cpu())

        # Plot training loss
        plot_loss(losses)

# Main execution block
if __name__ == "__main__":
    # Load dataset and prepare model
    train_loader, test_data = load_data(batch_size=BATCH_SIZE)
    model = SingleLayerNet(input_dim=64, output_dim=10)

    # Train the model
    losses = train_model(model, train_loader, lr=LEARNING_RATE, epochs=EPOCHS)

    # Evaluate on test data
    evaluate_model(model, test_data, losses)
