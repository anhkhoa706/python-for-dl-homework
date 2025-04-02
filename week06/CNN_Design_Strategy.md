
# CNN Design Strategy Guide

A practical guide for building effective Convolutional Neural Networks (CNNs), especially for image classification tasks like MNIST and CIFAR-10.

---

## 🏗️ Start Simple

Use a basic block structure:
```
Conv2D → BatchNorm → ReLU → MaxPool
```

Final layers:
```
Flatten → Dropout → Fully Connected → Output
```

---

## 📏 Manage Feature Map Size

- Start with 32 or 64 filters.
- Double the number of filters each block (32 → 64 → 128).
- Use MaxPooling (2×2) to reduce spatial dimensions gradually.

---

## ⚙️ Use BatchNorm and ReLU After Convolution

Batch normalization provides:
- Smoother gradients
- Faster training
- Less sensitivity to initialization

🔁 Recommended order:
```
Conv → BatchNorm → ReLU → MaxPool
```

---

## 🛡️ Apply Dropout After Flattening

Use Dropout in the fully connected layers to prevent overfitting:
```python
self.dropout = nn.Dropout(0.5)
```

Randomly disables neurons during training to promote generalization.

---

## 🔍 Use 3x3 Kernels with Padding

- Standard in modern CNNs (e.g., VGG, ResNet)
- Padding=1 preserves spatial size
- Efficient in both parameters and computation

### 🤔 Why 3x3 Instead of 5x5?

- 3x3 = 9 parameters vs 25 in 5x5 (more efficient)
- Two 3x3 layers approximate a 5x5 field with more non-linearity

📚 Reference: [VGGNet Paper](https://arxiv.org/abs/1409.1556)

---

## 📐 Flatten Before Dense Layers

Use `.view()` to reshape feature maps before feeding them into fully connected layers:
```python
x = x.view(x.size(0), -1)
```

Transforms shape `[batch_size, channels, height, width]` into `[batch_size, features]`.

---

## 🎯 Choose Output Layer Based on Task

| 🎯 Task Type            | 🧩 Output Layer                          |
|--------------------------|------------------------------------------|
| Classification           | `nn.Linear(..., num_classes)`           |
| Binary Classification    | `nn.Sigmoid()` or `Linear(..., 1)`      |
| Regression               | `nn.Linear(..., 1)`                     |

---

## 🧪 Sample Template: SimpleCNN in PyTorch

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
```

---

## 📚 References

1. [VGGNet Paper - Why 3x3 Convolutions](https://arxiv.org/abs/1409.1556)
2. [Batch Normalization Paper](https://arxiv.org/abs/1502.03167)
3. [Stanford CS231n Notes on CNNs](http://cs231n.stanford.edu/slides/2022/cs231n_2022_lecture5.pdf)
4. [PyTorch Official Documentation](https://pytorch.org/docs/stable/nn.html)
