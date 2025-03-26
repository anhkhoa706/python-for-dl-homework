import torch
import numpy as np
import matplotlib.pyplot as plt

# Set default figure size
plt.rcParams["figure.figsize"] = (9, 4)
# Default figure size
# plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]

# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Generate simulated data
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.rand(100, 1) * 0.1

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Initialize weights
w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

# Set learning rate and number of epochs
learning_rate = 0.01 # Adjusted learning rate
epochs = 100
losses = []

# Training loop
for epoch in range(epochs):
    # Predict using element-wise multiplication
    y_pred = X_tensor * w + b
    
    # Loss
    loss = torch.mean((y_pred - y_tensor) ** 2)
    losses.append(loss.item())
    
    # Gradients using autograd
    loss.backward()
    
    # Update weights
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
        # Zero gradients
        w.grad.zero_()
        b.grad.zero_()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Print trained weights
print(f'Trained weights: w = {w.item()}, b = {b.item()}')

# Plot data and fitted line
plt.subplot(1, 2, 1)
plt.scatter(X, y, label='Original data')
plt.plot(X, (w.detach().numpy() * X + b.detach().numpy()).flatten(), color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss Over Time')
plt.show()