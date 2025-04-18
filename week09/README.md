
# 🐶 Dog Breed Classification using EfficientNet & Vision Transformer

This project uses **transfer learning** with **EfficientNet** and **Vision Transformer (ViT)** to classify 8 dog breeds, based on data crawled from Bing.

---

## 📚 Dog Breed Labels (Based on AKC)

We used the [AKC Dog Breeds](https://www.akc.org/dog-breeds/) to define our breed list:
- Bichon Frise
- Bulldog
- Chihuahua
- Collie
- German Shepherd Dog
- Siberian Husky
- Pug
- Beagle

Each class has 150 images stored in:
```
data_dog_breeds/{Breed_Name}/image.jpg
```

---

## 🛠️ Project Features

- ✅ EfficientNet-B0 and ViT-B16 with pretrained ImageNet weights
- ✅ Custom dataset using `torchvision.datasets.ImageFolder`
- ✅ Train/Validation split
- ✅ TensorBoard logging (`runs/`)
- ✅ Val accuracy, confusion matrix, and prediction visualization
- ✅ Final model checkpoints saved to `checkpoints/`

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install torch torchvision matplotlib scikit-learn tensorboard
```

### 2. Prepare Dataset
Make sure your folder looks like this:
```
data_dog_breeds/
├── Beagle/
├── Bichon_Frise/
├── Bulldog/
├── Chihuahua/
├── Collie/
├── German_Shepherd_Dog/
├── Pug/
└── Siberian_Husky/
```

> You can crawl images with `icrawler`. Ask if you need a script for that.

---

### 3. Train Models & Visualize Training with TensorBoard
```bash
tranferlearning.ipynb
```
- Logs: `runs/efficientnet/`, `runs/vit/`
- Models: `checkpoints/efficientnet_model.pt`, `checkpoints/vit_model.pt`
- Outputs:  
  - `efficientnet_confusion_matrix.png`  
  - `efficientnet_Sample_Predictions.png`  
---

## 📈 Example Output

- Confusion matrix (8 classes)
- Predicted vs. ground truth visualization (first 8 validation images)
- TensorBoard logs for accuracy/loss

---

## 📂 File Structure

```
.
├── tranferlearning.ipynb   #Training script with TensorBoard and visualization              
├── data_dog_breeds/        # Dog image folders (one per breed)
├── checkpoints/            # Trained .pt models
├── runs/                   # TensorBoard logs
└── README.md               # You are here
```

---

## 📌 Credits

- AKC Dog Breeds: https://www.akc.org/dog-breeds/
- Models from `torchvision.models`
- Data crawled with `icrawler` (optional)

---
