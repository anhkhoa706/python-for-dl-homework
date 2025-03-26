import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Training parameters
LEARNING_RATE = 1e-3  # Adjusted for better convergence
EPOCHS = 40_000       
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
DATA_PATH = config["data_path"]

# Define the model
class PricePredictor(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def normalize_data(tensor: torch.Tensor) -> torch.Tensor:   
    """Normalizes a tensor using mean and standard deviation."""
    return (tensor - tensor.mean()) / tensor.std()

def load_data(file_path: str):
    """
    Load and preprocess CSV data.
    """
    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Define feature and target columns
    feature_columns = [
        'main building area', 'auxiliary building area', 'balcony area', 
        'property age', 'distance to city center', 'area (square meters)'
    ]
    target_column = 'total price NTD'

    # Extract and convert data
    features = torch.tensor(df[feature_columns].values, dtype=torch.float32)
    target = torch.tensor(df[target_column].values, dtype=torch.float32).view(-1, 1)

    # Normalize features
    features = normalize_data(features)

    return features.to(DEVICE), target.to(DEVICE)

def train_model(model: torch.nn.Module, features: torch.Tensor, target: torch.Tensor, 
                epochs: int, learning_rate: float):
    """
    Train the model using MSE loss and SGD optimizer.
    """
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_fn(predictions, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    return losses

def evaluate_model(model: torch.nn.Module, features: torch.Tensor, target: torch.Tensor):
    """
    Evaluate and visualize predictions vs actual values.
    """
    model.eval()  
    with torch.no_grad():
        predictions = model(features).cpu().numpy()
        actual = target.cpu().numpy()

    # Compute Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predictions, alpha=0.6, color='blue', label="Predictions")
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2, label="Perfect Fit")
    plt.xlabel("Actual Total Price")
    plt.ylabel("Predicted Total Price")
    plt.title("Actual vs Predicted Total Price")
    plt.legend()
    plt.show()

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

def main():
    # Load data
    data_file = os.path.join(DATA_PATH, "processed", "selected_columns.csv")
    features, target = load_data(data_file)

    # Initialize model
    model = PricePredictor(features.shape[1]).to(DEVICE)

    # Train the model
    losses = train_model(model, features, target, EPOCHS, LEARNING_RATE)

    # Display trained parameters
    weights = model.linear.weight[0].tolist()
    bias = model.linear.bias.item()
    print(f'Final Weights: {weights}')
    print(f'Final Bias: {bias:.4f}')
    
    # Plot training loss
    plot_loss(losses)

    # Evaluate model
    evaluate_model(model, features, target)

if __name__ == '__main__':
    main()
