# Diabetes Prediction using Machine Learning

## Overview

This project develops a machine learning model for predicting diabetes using patient health indicators.

The project follows a complete Data Science workflow including:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Model Training
- Hyperparameter Optimization
- Model Evaluation
- Experiment Tracking with MLflow
# Start MlFlow / You can check all experiments
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

The objective is to build a reliable binary classification model capable of identifying patients at risk of diabetes while maintaining a good balance between precision and recall.

---

## Dataset

The project uses publicly available diabetes datasets containing demographic and medical information such as:

- Age
- Gender
- BMI
- Hypertension
- Heart disease
- Smoking history
- HbA1c level
- Blood glucose level

Target variable:

- **diabetes**
    - 0 → No diabetes
    - 1 → Diabetes

---

## Project Structure

```
SU_ML_RETAKE_EXAM_2026
│
├── data
│   ├── raw
│   └── processed
│
├── models
│
├── notebooks
│   ├── 01_overview.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
│   └── 05_alternative_modeling.ipynb
│
├── reports
│   └── figures
│
├── source
│   ├── data
│   ├── features
│   ├── helper
│   ├── models
│   └── visualization
│
└── README.md
```

---

## Workflow

### 1. Exploratory Data Analysis

The dataset was analyzed to understand:

- missing values
- class distribution
- feature distributions
- correlations
- outliers
- relationships between medical variables

Visualizations include:

- Histograms
- Correlation matrices
- Q-Q plots
- Class distributions

---

### 2. Feature Engineering

Several preprocessing steps were applied:

- One-Hot Encoding for categorical variables
- Standardization of numerical features
- Dataset transformation using Scikit-Learn pipelines
- Train / Validation split

---

### 3. Modeling

The project uses a **Random Forest Classifier**.

Hyperparameters were optimized using **GridSearchCV**.

Several combinations of:

- number of trees
- tree depth
- minimum samples
- class weights

were evaluated to improve the model's ability to detect diabetes cases.

---

### 4. Threshold Optimization

Instead of using the default probability threshold (0.50), multiple decision thresholds were evaluated.

This allows improving recall for the minority class while controlling the number of false positives.

---

### 5. Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Feature Importance

Special attention was given to Recall because correctly identifying diabetic patients is more important than maximizing overall accuracy.

---

## Experiment Tracking

MLflow was used for:

- experiment logging
- hyperparameter tracking
- metric comparison
- model artifact storage

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- MLflow
- Jupyter Notebook

---

## Results

The final Random Forest model achieved strong overall performance while maintaining good generalization on both the test and validation datasets.

Threshold tuning further improved the model's ability to identify diabetic patients by increasing recall for the positive class.

Feature importance analysis also showed that medical indicators such as HbA1c level and blood glucose level have the greatest influence on model predictions.

### Experiment 3

### Classification Report

The confusion matrix summarizes the classification performance of the final Random Forest model. It shows the number of correctly and incorrectly classified instances for both classes. The model achieves excellent performance on the majority class (healthy patients) while maintaining a strong recall for the minority class (diabetic patients).

This is the best-performing model obtained throughout the experiments. Considering the class imbalance, the achieved results demonstrate a good balance between overall accuracy and the ability to detect diabetic patients. However, further improvements are likely to require a larger and more balanced dataset. Increasing the number of positive (diabetes) cases would provide the model with more representative examples, helping it learn more robust decision boundaries and further reduce the number of False Negatives.

| Class | Precision | Recall | F1-score | Support |
|------:|----------:|-------:|---------:|--------:|
| 0 (Healthy) | 0.98 | 0.97 | 0.98 | 14,640 |
| 1 (Diabetes) | 0.72 | 0.80 | 0.76 | 1,360 |


![Experiment 3 - Confusion metrix](reports/figures/experiment_3_025_th.png)

### Working with CatBoost Model to Check the Results / 05_alternative_modeling.ipynb

In this part of the notebook, we trained a CatBoost classification model to evaluate how well it performs on the diabetes prediction task.

We first initialized the model with a minimal set of parameters in order to establish a baseline before further hyperparameter tuning.

The main focus was not only overall accuracy, but especially the model’s ability to correctly detect patients with diabetes. Since **Class 1 represents diabetic patients**, **Recall for Class 1** was treated as the most important metric.

The model was evaluated on both the validation and test datasets. The results showed a clear trade-off between **Precision** and **Recall**. On the validation dataset, the model achieved very high precision for Class 1, but missed a larger number of diabetic patients. On the test dataset, the recall for Class 1 improved, meaning the model detected more diabetic patients, although this came with more false positive predictions.

Overall, the CatBoost model showed promising results for this classification task. Since the main goal is to minimize false negatives, the version with higher recall for Class 1 is more suitable for the project objective.

![Alternative modeling with Catbookst model](reports/figures/catboost_results.png)