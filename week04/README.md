
# 🧠 Simple Perceptron: Scikit-learn & PyTorch

## 📘 Project Overview

This project demonstrates how to implement a basic **Perceptron neural network** using both:

- ✅ **Scikit-learn**: for a quick and high-level API experience.
- 🔧 **PyTorch**: for hands-on, manual implementation to build deep understanding.

It is designed for educational purposes to help learners grasp fundamental neural network concepts, explore training behavior, and understand the effect of data normalization and activation functions.

---

## 🗂️ Project Structure

```
PERCEPTRON/
│
├── legacy-src/                        # Old or experimental versions
│
├── sklearn_perceptron.py              # Basic scikit-learn implementation (script)
├── sklearn_perceptron_scaled.py       # Scikit-learn with StandardScaler applied
├── sklearn_perceptron.ipynb           # Notebook version with evaluation and plots
│
├── pytorch_perceptron.py              # Manual Perceptron implementation using PyTorch
├── simple_perceptron.py               # A basic perceptron logic implementation
├── activation_functions_demo.ipynb    # Visualizing and testing activation functions
│
└── README.md                          # This file
```

---

## 🚀 How to Run

### ✅ Scikit-learn Version

```bash
python sklearn_perceptron.py
```

For normalized version:
```bash
python sklearn_perceptron_scaled.py
```

Or use the notebook:
```bash
jupyter notebook sklearn_perceptron.ipynb
```

### 🔧 PyTorch Version

Make sure PyTorch is installed:
```bash
pip install torch
```

Then run:
```bash
python pytorch_perceptron.py
```

---

## 📊 Results Overview

| Model                | Normalization | Accuracy |
|---------------------|---------------|----------|
| Scikit-learn        | No            | 0.92     |
| Scikit-learn        | Yes           | **0.95** |
| PyTorch (Manual)    | Yes           | 0.93–0.95|

---

## 📚 Learning Objectives

- Understand how a perceptron works
- Compare high-level (scikit-learn) vs low-level (PyTorch) training approaches
- Observe how **hyperparameters** (like learning rate and epochs) impact accuracy
- Explore how **data normalization** improves model performance
- Visualize **loss curves** and **confusion matrices**

---

## 🧠 Bonus: Activation Functions

The notebook `activation_functions_demo.ipynb` shows how different activation functions behave visually, including:
- Sigmoid
- Tanh
- ReLU

---

## 🙌 Acknowledgements

Built as a learning exercise to understand core concepts in neural networks using hands-on coding.

---

## 🔄 To-Do (Optional Enhancements)

- [ ] Add MLP (multi-layer perceptron)
- [ ] Use MNIST dataset (28x28) instead of 8x8 digits
