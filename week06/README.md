# Improve CNN Model with TensorBoard (EMNIST, MNIST, FashionMNIST, KMNIST)

This repository contains multiple neural network models to classify handwritten digits and characters using PyTorch and visualize performance using TensorBoard.

## 📌 Objectives

- Improve accuracy of a basic CNN model
- Apply model to multiple datasets: **MNIST**, **FashionMNIST**, **KMNIST**, and **EMNIST (balanced)**
- Visualize training metrics with **TensorBoard**
- Understand how CNNs learn and where they make mistakes

## 📁 File Descriptions

| File Name                    | Description |
|-----------------------------|-------------|
| `MLP_3layer_Legacy_MNIST.ipynb`        | Basic 3-layer fully connected NN |
| `CNN_Legacy_SingleConv_MNIST.ipynb`    | Original CNN with 1 conv layer |
| `CNN_Improved_MNIST.ipynb`             | Better CNN architecture with 3 conv layers, batch norm, dropout |
| `CNN_MNIST_KMNIST_Fashion_EMNIST.ipynb`| Modular training for 4 datasets + TensorBoard + visualization |

## 📊 Visualizations

- Real-time tracking with **TensorBoard**: ```tensorboard --logdir=runs```
- Misclassified test samples shown with:
- Prediction (P)
- Ground Truth (GT)
- Class names (especially for EMNIST)

## 🧪 Datasets Used

All datasets are loaded via `torchvision.datasets`:

- `MNIST`: Digits (0–9)
- `FashionMNIST`: Clothing categories
- `KMNIST`: Japanese characters (hiragana)
- `EMNIST (balanced)`: 47-class mix of letters and digits

## 🎞️ Visual Explanation

Understand how a Convolutional Neural Network works visually:

📺 [Watch: How CNNs work (YouTube)](https://www.youtube.com/watch?v=eMXuk97NeSI)
🧠 [Design CNN](./CNN_Design_Strategy.md)

## 🧰 How to Use

1. Install dependencies:
 ```bash
 pip install torch torchvision tensorboard matplotlib
 ```
2. Run any .ipynb notebook in Jupyter or VSCode
3. Launch TensorBoard to track metrics: