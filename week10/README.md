# 📊 Crowd Counting

This project builds a simple yet complete **crowd counting system** using deep learning, based on fine-tuning lightweight pre-trained models.

## 🌍 Objective

Achieve **the best model** balancing **accuracy**, **small model size**, and **efficient computation**.

We:
- Selected ShuffleNet V2 0.5× backbone for efficiency.
- Fine-tuned it for **regression-based people counting**.
- Applied augmentation and CutMix to boost generalization.

## 🔄 Project Flow

- **Notebook**: `week10_crowd_counting.ipynb`
- **Key Components**:
  - Data Loading and Preprocessing
  - Model Architecture Definition (ShuffleNet V2 0.5×)
  - Data Augmentation (ColorJitter, Random Grayscale, CutMix)
  - Training and Loss Monitoring
  - Model Evaluation (MAE, Accuracy, Visualization)
  - Final Model Score Calculation

✅ We selected ShuffleNet V2 0.5× because after using `crawl_weight.py` to crawl the table of torchvision pre-trained models, ShuffleNet achieved **Rank 1** for the best trade-off between model size, GFLOPS, and initial accuracy.

## 🌐 Model Score Calculation

The **Model Score** formula:

```
Model Score = (1 - accuracy%) × (number of parameters in M) × GFLOPS × (number of training images)
```

We optimized:
- **Accuracy**: Maximize prediction performance.
- **Efficiency**: Minimize model parameters and GFLOPS.

✅ A balance between **small model**, **good accuracy**, and **low computation**.

---

# 🧪 Final Results

| Method | CutMix Alpha | Augmentation | MAE ↓ | Accuracy ↑ |
|:--|:--|:--|:--|:--|
| No Augment | - | ❌ | 118.23 | 72.73% |
| Augment Only | - | ✅ | 117.81 | 72.85% |
| CutMix 0.8 + Augment | 0.8 | ✅ | 117.64 | 72.89% |
| CutMix 1.0 + Augment (**Best Run**) | 1.0 | ✅ | **112.11** | **74.16%** |
| CutMix 1.5 + Augment | 1.5 | ✅ | 115.47 | 73.39% |
| CutMix 2.0 + Augment | 2.0 | ✅ | 121.29 | 72.05% |

✅ **Best configuration** achieved **MAE 112.11** and **Accuracy 74.16%**.

---

# 📂 How to Run

### 1. Install Required Libraries

```bash
pip install torch torchvision matplotlib ptflops
```

### 2. Open and Execute the Notebook

```bash
jupyter notebook week10_crowd_counting.ipynb
```

Follow each cell:
- Load training data (aerial crowd images + people counts).
- Preprocess images (resize, normalize).
- Load a pre-trained model from [torchvision models](https://pytorch.org/vision/stable/models.html).
- Modify the model for regression (predict a number instead of classification).
- Train and record the loss.
- Evaluate and visualize predictions.

Model and TensorBoard logs are automatically saved.

---

# 🎨 Dataset

- **Dataset**: Aerial images of crowds from [ShanghaiTechDataset](https://github.com/desenzhou/ShanghaiTechDataset)
- **Input**: Crowd images
- **Label**: Ground-truth number of people (loaded via `.mat` files)

Preprocessing includes resizing, normalization, and optional augmentation.

---

# 🏋️‍♂️ Model Details

- **Architecture**: ShuffleNet V2 0.5× (pre-trained on ImageNet)
- **Modification**: Final fully-connected layer adjusted for single regression output
- **Loss Function**: Smooth L1 Loss (Huber Loss)
- **Optimizer**: AdamW
- **Scheduler**: OneCycleLR
- **Data Augmentation**:
  - Random Color Jitter
  - Random Grayscale
  - Random Rotation
  - Gaussian Blur
  - **CutMix Augmentation**

---

# 🔍 Key Observations

- CutMix augmentation **significantly improves** counting accuracy.
- Optimal CutMix Alpha: **1.0**.
- Data augmentation alone also helps, but less than CutMix.

---

# ✨ Strategic Tips

> 🔗 Smaller + Smarter + More Accurate = Higher Score!

- Always balance **model size** and **accuracy**.
- Proper **augmentation** boosts real-world generalization.
- Fine-tuning with a lightweight backbone like ShuffleNet V2 achieves excellent efficiency.

---

# 📈 TensorBoard Support

You can run TensorBoard for live training tracking:

```bash
tensorboard --logdir=runs --port=6009
```

---

✅ Completed with full experiments, ablation studies, and final model optimization!