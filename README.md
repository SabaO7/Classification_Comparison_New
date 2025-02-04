# README: Text Classification Framework

## Overview

This project implements a comprehensive framework for text classification tasks using multiple classifiers and utilities. It is designed to handle various stages of data processing, model training, evaluation, and debugging. The framework supports both traditional and transformer-based classifiers with features such as cross-validation, dynamic data loading, and metrics visualization.

### Key Features:

- **Support for multiple classifiers**: BERT-based classifier, logistic regression, and few-shot learning.
- **Preprocessing utilities**: Tools for pre-tokenization and data preparation.
- **Performance metrics**: Evaluation using accuracy, precision, recall, F1 score, and ROC-AUC.
- **Visualization**: Tools for visualizing performance metrics.
- **Debugging**: Scripts to inspect datasets and troubleshoot model training.
- **Modular design**: Easy to extend and integrate additional classifiers or utilities.

---

## Project Structure

```
project/
├── classifiers/
│   ├── bert_classifier.py         # BERT-based text classifier
│   ├── logistic_classifier.py     # Logistic regression classifier
│   ├── few_shot_classifier.py     # Few-shot learning classifier
│   ├── debug_bert.py              # Debugging script for BERT classifier
├── utils/
│   ├── data_preprocessing.py      # Pre-tokenization and preprocessing
│   ├── performance_metrics.py     # Metrics calculation and evaluation
│   ├── visualization.py           # Visualization of metrics
│   ├── base_classes.py            # Base classes for shared functionality
├── data/
│   ├── raw/                       # Raw datasets
│   ├── tokenized/                 # Tokenized datasets
├── outputs/
│   ├── logs/                      # Logs for training and evaluation
│   ├── models/                    # Saved model checkpoints
├── main.py                        # Entry point for running the classifiers
├── requirements.txt               # Python dependencies
├── README.md                      # Documentation
└── .env                           # Environment variables
```

---

## Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

### Key Libraries:

- `transformers`: For BERT-based models and tokenizers.
- `torch`: PyTorch for model training and dataset handling.
- `sklearn`: For logistic regression and evaluation metrics.
- `pandas`: For data manipulation.
- `matplotlib`: For visualizations.

---

## Classifiers

### 1. **BERTClassifier**

A transformer-based classifier leveraging the Hugging Face `transformers` library. Designed for large-scale datasets with features like cross-validation and dynamic data loading.

#### Key Methods:

- **Initialization**: Sets up tokenizer, model, and device (CPU/GPU).
- **train_fold**: Trains the model using a fold of data in cross-validation.
- **evaluate_model**: Evaluates the trained model on the test set.
- **save_model**: Saves the trained model and tokenizer.
- **compute_metrics**: Computes accuracy, precision, recall, F1, and ROC-AUC.

### 2. **LogisticClassifier**

A traditional machine learning classifier implemented using `scikit-learn`.

#### Key Methods:

- **train**: Trains a logistic regression model.
- **evaluate**: Computes metrics on the test set.
- **save_model**: Saves the trained logistic regression model.

### 3. **FewShotClassifier**

A few-shot learning implementation for low-resource text classification tasks.

#### Key Methods:

- **train**: Fine-tunes the model with a small labeled dataset.
- **evaluate**: Evaluates the few-shot model on unseen data.
- **predict**: Makes predictions on new samples.

---

## Utilities

### 1. **Data Preprocessing**

- **data_preprocessing.py**: Handles raw dataset tokenization and saves the tokenized output for efficient training.

### 2. **Performance Metrics**

- **performance_metrics.py**: Computes and aggregates metrics for evaluation.

### 3. **Visualization**

- **visualization.py**: Plots performance metrics for comparative analysis.

### 4. **Base Classes**

- **base_classes.py**: Provides reusable base classes for classifiers and configurations.

---

## Workflow

### 1. **Prepare the Dataset**

- Place raw data in `data/raw/`.
- Run the preprocessing script:

```bash
python utils/data_preprocessing.py
```




# Suicide Detection Classification Framework

## Overview
This project implements **three different classifiers** to detect suicide-related text in a dataset. The classifiers are:

1. **Logistic Regression** (TF-IDF based)
2. **BERT Fine-Tuning**
3. **Few-Shot Learning with GPT-4o**

Each classifier is evaluated based on **accuracy, precision, recall, F1-score, and ROC-AUC** to compare their performance.

---

## Training & Evaluation Process

### 1. Logistic Regression & BERT Classifiers
#### Training and Evaluation Process
- **80/20 Split**
  - **80% of the dataset** is used for training.
  - **20% of the dataset** is kept as the final test set (**never seen during training**).

- **5-Fold Cross-Validation on the 80% Training Set**
  - The **80% training data** is **split into 5 equal folds**.
  - The model is trained **5 times**, each time:
    - **Using 4 folds (64% of original dataset) for training.**
    - **Using 1 fold (16% of original dataset) for validation.**
  - The **best-performing fold is selected**.

- **Final Model Training on the Full 80% Training Data**
  - The model is **retrained using all 80% of the data** (not just 4 folds).
  - This final model is **not validated again**—it's ready for testing.

- **Final Testing on the 20% Test Set**
  - The final trained model is **evaluated once on the 20% test set**.
  - **Performance metrics** (accuracy, F1-score, precision, recall, ROC-AUC) are calculated.

✅ **This is the same process for both Logistic Regression and BERT.**

---

### 2. Few-Shot Learning (GPT-4o)
🚨 **The Few-Shot Classifier does NOT follow the same 80/20 + 5-Fold approach.**

#### What’s Actually Happening in `few_shot_classifier.py`
- **The Model Does NOT Train on 80% of the Data:**
  - Instead of using 80% of the dataset for training, Few-Shot Learning **only uses a few predefined examples** (hardcoded in the script):
    ```python
    self.examples = [
        {"text": "I feel like cutting myself.", "label": "suicide"},
        {"text": "I am going to the gym.", "label": "non-suicide"},
        {"text": "I want to hurt myself.", "label": "suicide"},
        {"text": "I am feeling happy today.", "label": "non-suicide"},
    ]
    ```
  - These **4 manually defined examples** act as "training data."

- **Directly Uses GPT-4o to Classify the 20% Test Set**
  - The model then takes the **20% test set** and classifies each text **based on the few-shot examples**.
  - 🚨 **GPT-4o never sees the 80% training set.**
  - Instead, it **relies only on the given few-shot examples** to make predictions.

- **Performance Metrics Are Calculated on the 20% Test Set**
  - After classifying the **20% test set**, performance is evaluated using:
    ```python
    val_preds = self.classify_batch(X_val)
    metrics = {
        'accuracy': accuracy_score(y_val, val_preds),
        'precision': precision_score(y_val, val_preds, pos_label='suicide'),
        'recall': recall_score(y_val, val_preds, pos_label='suicide'),
        'f1': f1_score(y_val, val_preds, pos_label='suicide'),
        'roc_auc': roc_auc_score((y_val == 'suicide').astype(int), val_probs)
    }
    ```
  - ✅ **GPT-4o’s performance is only evaluated on the 20% test set.**

---

## Cross-Validation and Final Model Training Process
This applies to **Logistic Regression & BERT Classifiers**, but NOT Few-Shot Learning.

1. **Train-Test Split (80-20)**
   - The dataset is first split into **80% training** and **20% testing**.
   - The **test set (20%) remains untouched** until the final evaluation.

2. **5-Fold Cross-Validation on the 80% Training Data**
   - The **80% training data** is further split into **5 equal parts (folds)**.
   - The model is trained **5 times**, each time:
     - **4 folds (64% of total data) are used for training.**
     - **1 fold (16% of total data) is used for validation.**
   - This ensures that every data point is used for **validation once** and **training multiple times**.

3. **Selecting the Best Model from Cross-Validation**
   - After 5-fold CV, the **best-performing fold** is selected based on metrics like **F1-score, accuracy, precision, recall, and ROC-AUC**.
   - The **best model configuration** (hyperparameters, training setup) is recorded.

4. **Final Training on the Entire 80% Training Data**
   - Instead of training on only 4 folds, the **final model is trained using the full 80% training data**.
   - This **maximizes the available training data** to improve generalization.

5. **Final Model Evaluation on the 20% Test Set**
   - The **final trained model is tested ONCE on the 20% test set**.
   - This provides an **unbiased** estimate of real-world performance.
   - The test set **was never used in training or validation**, ensuring a **fair evaluation**.

### Why This Process?
✅ Prevents **data leakage** by keeping the test set separate.  
✅ Ensures **comparability** between different models.  
✅ Maximizes the **amount of training data** before final evaluation.  
✅ Provides a **robust estimate** of real-world performance.  

---

## Next Steps: Fixing RAM Issues in Data Preprocessing

### 🚨 Why Is RAM Usage High?
Even though **only 10% of the dataset** is supposed to be used, the code is still processing **the full dataset** in some cases, leading to excessive RAM consumption.

- **Multiple copies of DataFrames are created at each preprocessing step.**
- **Tokenization (for BERT) and TF-IDF Vectorization (for Logistic Regression) are memory-intensive.**
- **Parallel API calls for Few-Shot Learning store intermediate results in RAM.**
- **Mac M1 (8GB RAM) struggles with large Pandas DataFrames and Torch tensors.**

### 🛠️ How to Fix It?
1. **Explicitly enforce dataset sampling at the beginning of preprocessing:**
   ```python
   df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)  # Ensure only 10% is used
   ```
2. **Modify DataFrames in place to reduce memory consumption:**
   ```python
   df.dropna(subset=['text'], inplace=True)
   df.drop_duplicates(subset=['text'], inplace=True)
   ```
3. **Reduce batch size in BERT training to lower GPU/CPU memory load.**
4. **Reduce TF-IDF max features to minimize vectorization memory.**
5. **Limit Few-Shot Learning API calls to prevent excessive memory usage.**

---

## Conclusion
- **Logistic Regression & BERT Classifiers follow 80/20 + 5-Fold CV.**
- **Few-Shot Learning only uses predefined examples and skips training.**
- **RAM issues arise due to redundant DataFrames and large processing steps.**
- **Optimizations will improve performance and prevent Mac M1 crashes.**

