# 📦 Week 8 - Lightweight CNN Model

This repository contains a **lightweight Convolutional Neural Network (CNN)** model designed for classifying handwritten and fashion-related character datasets. Supported datasets include:

- 🖊️ MNIST
- 👗 Fashion-MNIST
- 🈶 KMNIST

## 📁 File Description

- `cnn_mnist_fashion_kmnist.ipynb`: Jupyter notebook implementing the training and evaluation of a compact CNN architecture.

## ✨ Features

- ✅ Depthwise Separable Convolutions for parameter efficiency
- 🔍 SE (Squeeze-and-Excitation) blocks for channel attention
- 🔧 Adjustable number of channels for dataset-specific performance
- 📊 Evaluation on multiple datasets
- 📉 TensorBoard support for training visualization

## 🚀 How to Run

1. Install the required packages:
```bash
pip install torch torchvision matplotlib
```

2. Launch Jupyter Notebook and open:
```
cnn_mnist_fashion_kmnist.ipynb
```

3. Choose the dataset and run all cells to train and evaluate the model.

## 🎥 Related Video

Watch the concept explanation for **Groups, Depthwise, and Depthwise-Separable Convolution (Neural Networks)** here:  
👉 [YouTube Link](https://www.youtube.com/watch?v=vVaRhZXovbw)

---

Enjoy exploring lightweight deep learning! 🔬💡
