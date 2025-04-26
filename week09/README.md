# 🐶 Dog Breed Classification via Transfer Learning

This project classifies **12 dog breeds** using **transfer learning** with four deep learning models: GoogLeNet, ResNet-18, EfficientNet-B0, and Swin Transformer.

---

## 📚 Dog Breeds

Collected via Bing Search:

- Beagle, Bichon Frise, Bulldog, Chihuahua, Collie
- German Shepherd, Pug, Siberian Husky, Dalmatian
- Shiba Inu, Yorkshire Terrier, Labrador Retriever

Each breed has ~110-125 images under:
```
data_dog_breeds/{Breed_Name}/image.jpg
```

---

## 🛠️ Features

- Custom dataset cleaned & validated
- Strong data augmentation
- Transfer learning with 4 pretrained models
- Dropout regularization before classifier head
- TensorBoard logging, confusion matrices, learning curves
- Checkpoints saved after training

---

## 🚀 Quick Start

1. Install dependencies:
```bash
pip install torch torchvision matplotlib scikit-learn tensorboard
```

2. Folder Structure:
```
data_dog_breeds/{Breed}/
```

3. Train & Evaluate:
```bash
transferlearning.ipynb
```

TensorBoard:
```bash
tensorboard --logdir=runs
```

---

## 📊 Model Summary

| Model              | Params (M) | Best Val Accuracy |
|:-------------------|:----------:|:-----------------:|
| EfficientNet-B0    | 4.02M       | ~96.1%            |
| GoogLeNet          | 5.61M       | ~95.8%            |
| ResNet-18          | 11.18M      | ~96.8%            |
| Swin Transformer   | 27.53M      | ~99.3%            |

---

## 📂 Files

```
.
├── transferlearning.ipynb   # Training notebook
├── data_dog_breeds/         # Dataset
├── checkpoints/             # Saved models
├── runs/                    # TensorBoard logs
└── README.md
```

---

## 📌 Credits

- Dataset inspired by AKC Dog Breeds
- Pretrained models from torchvision
- Crawling using icrawler

