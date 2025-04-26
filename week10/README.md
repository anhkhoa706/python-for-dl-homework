# 📊 Crowd Counting

This project builds a simple yet complete **crowd counting system** using deep learning, based on selecting a pre-trained model and predicting the number of people in images.

## 🌍 Objective

Achieve **the best result** using the **smallest model** with the **fewest resources**! 
You will:
- Select a pre-trained model.
- Fine-tune it to **predict the number of people**.
- Minimize your **Model Score** by balancing **accuracy**, **model size**, and **computation cost**.

## 🔄 Project Flow

- **Notebook**: `week10_crowd_counting.ipynb`
- **Main Sections**:
  - Data Loading and Preprocessing
  - Model Architecture Definition
  - Model Fine-tuning for Regression
  - Training and Loss Monitoring
  - Model Evaluation and Visualization

## 🌐 Model Score Calculation

You must calculate and **show** your model score using:

```
Model Score = (1 - accuracy%) × (number of parameters in M) × GFLOPS × (number of training images)
```

- **Lower Model Score = Better Ranking** 📈
- After calculating:
  - 1st place = 100 points
  - 2nd place = 95 points
  - 3rd place = 90 points
  - and so on...

✅ Make your model **small**, **accurate**, and **efficient**.


## 📂 How to Run

### 1. Install Required Libraries

```bash
pip install torch torchvision matplotlib
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

## 🎨 Dataset

- **Dataset**: Aerial images of crowds from [ShanghaiTechDataset](https://github.com/desenzhou/ShanghaiTechDataset)
- Ground-truth labels are loaded using `week10_2_MatParser.ipynb`.
- Each sample:
  - **Input**: A crowd image
  - **Label**: Number of people

## 🏋️ Model Details

- **Input**: Crowd images
- **Output**: Predicted number of people
- **Architecture**: Pre-trained CNN fine-tuned for regression
- **Loss Function**: Smooth L1 Loss (Huber Loss)
- **Optimizer**: Adam

## 🔍 Outputs

- A trained regression model.
- Prediction visualization (scatter plot or direct comparison).
- Final **Model Score** (MUST be reported).

## ✨ Key Strategy Tips

- Choose a model with **few parameters and low GFLOPS**.
- Train efficiently without overfitting.
- Prioritize **accuracy** but balance **efficiency**.
- Visualize results clearly.

> 🔗 Remember: Smaller + Smarter + More Accurate = Higher Score!

