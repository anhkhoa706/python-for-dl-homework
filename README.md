# Python for Deep Learning - Homework Repository

Welcome to the **Python for Deep Learning (DL)** course homework repository!  
This repository is organized week-by-week to showcase projects and hands-on exercises developed throughout the course.

---

## 📅 Weekly Homework Overview

### 📘 Week 01–02: Maze Game  
A Python-based interactive **Maze Game** built with Tkinter.  
- Procedural maze generation  
- Manual or auto-solve navigation  
- GUI with arrow-key support  
- 📄 See: `week01-02/README.md`

---

### 🏠 Week 03: Real Estate Price Prediction  
A linear regression model using **PyTorch** to predict housing prices based on features such as property age, area, and distance to the city center.  
- Trained using stochastic gradient descent  
- Includes training curve and MAPE evaluation  
- 📄 See: `week03/README.md`

---

### 🧠 Week 04: Perceptron Implementation  
Implementation of a **simple perceptron** using:  
- `scikit-learn` for baseline experimentation  
- `PyTorch` for custom hands-on model  
- Comparison of normalization, loss curves, and confusion matrices  
- 📄 See: `week04/README.md`

---

### 🌀 Week 06: CNN Improvements Across Datasets  
Enhanced a **convolutional neural network (CNN)** and applied it to MNIST, FashionMNIST, and EMNIST datasets.  
- Improved architecture with multiple convolutional layers, batch normalization, and dropout  
- Training/validation metrics and misclassified samples visualized using TensorBoard  
- 📄 See: `week06/README.md`

---

### ⚡ Week 08: Lightweight CNN Model  
Developed a **lightweight CNN architecture** and applied it to MNIST, FashionMNIST, KMNIST, and EMNIST.  
- Utilized depthwise separable convolutions and Squeeze-and-Excitation blocks  
- Incorporated data augmentation, learning rate scheduling, and early stopping  
- Configured `num_workers` for efficient data loading during GPU training  
- Visualized model performance and misclassifications via TensorBoard  
- 📄 See: `week08/README.md`

---

### 🐶 Week 09: Dog Breed Classification
Applied transfer learning to classify 12 dog breeds using GoogLeNet, ResNet-18, EfficientNet-B0, and Swin Transformer.  
- Dataset collected via Bing Search  
- Aggressive data augmentation and dropout regularization  
- TensorBoard logging and confusion matrices for evaluation  
- 📄 See: `week09/README.md`

---

### 📊 Week 10: Crowd Counting
Built a crowd counting regression model using a fine-tuned pre-trained CNN.  
- Dataset: Aerial crowd images from ShanghaiTechDataset  
- Fine-tuned with Smooth L1 Loss (Huber Loss) and evaluated with custom Model Score  
- Emphasis on balancing accuracy, model size, and efficiency  
- 📄 See: `week10/README.md`

---

## 🚀 How to Use

1. Clone the repository:
```bash
git clone <your-repo-url>
cd python-for-dl-homework
```

2. Navigate to a specific week's folder and follow its instructions:
```bash
cd week03
python price_prediction.py
```

---

## 📜 License  
All content in this repository is licensed under the [MIT License](LICENSE).  
You are welcome to explore, learn from, and build upon this work.