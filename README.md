# README: Suicide Detection Classification Framework

## Overview
This project implements a **multi-classifier text classification framework** for detecting suicide-related content in text data. It includes:

1. **Logistic Regression Classifier** (TF-IDF based)
2. **BERT Fine-Tuned Classifier**
3. **Few-Shot Learning with gpt-3.5-turbo**

The framework is designed to **handle preprocessing, model training, evaluation, debugging, and visualization** with modularity and scalability in mind.

---

## Key Features
- **Multi-Model Classification**: Supports **Logistic Regression**, **BERT**, and **Few-Shot Learning**.
- **Preprocessing Pipeline**: Data cleaning, tokenization, and sampling.
- **Cross-Validation & Training**: **5-fold cross-validation** for model robustness.
- **Comprehensive Evaluation**: Metrics include accuracy, precision, recall, F1-score, and ROC-AUC.
- **Performance Tracking**: Logs execution time and GPU usage.
- **Visualization**: ROC curves, confusion matrices, and performance comparisons.
- **Memory Optimization**: Prevents excessive RAM consumption by early dataset sampling and in-place operations.

---

## Project Structure
```
classification_framework/
├── data/
│   ├── raw/                        # Raw datasets
│   │   ├── Suicide_Detection.csv    # Main dataset
│   ├── tokenized/                   # Tokenized datasets for BERT
│   │   ├── tokenized_data.pkl       # Pre-tokenized dataset
├── src/
│   ├── classifiers/
│   │   ├── bert_classifier.py       # BERT-based classifier
│   │   ├── logistic_classifier.py   # Logistic regression classifier
│   │   ├── few_shot_classifier.py   # Few-shot learning classifier
│   │   ├── debug_bert.py            # Debugging script for BERT classifier
│   ├── utils/
│   │   ├── data_preprocessing.py    # Data cleaning and tokenization
│   │   ├── pre_tokenization_bert.py # Pre-tokenization for BERT
│   │   ├── performance_metrics.py   # Execution time & memory tracking
│   │   ├── visualization.py         # Plotting metrics and confusion matrices
│   ├── outputs/
│   │   ├── logs/                    # Logs for training and evaluation
│   │   ├── models/                  # Saved models
│   ├── main.py                      # Pipeline entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # Documentation
└── .env                              # Environment variables
```

---

## Installation
To install required dependencies, run:
```bash
pip install -r requirements.txt
```
### Key Libraries:
- `transformers`: For BERT-based models.
- `torch`: For deep learning.
- `sklearn`: For logistic regression and evaluation metrics.
- `pandas`: For data handling.
- `matplotlib & seaborn`: For visualizations.

---

## Classifiers

### 1. **Logistic Regression Classifier**
A traditional ML classifier using **TF-IDF vectorization** with **balanced class weights**.
#### Key Methods:
- **train_fold**: Performs **5-fold cross-validation**.
- **train_final_model**: Trains on the entire dataset.
- **evaluate_model**: Computes final metrics on the test set.

### 2. **BERT Fine-Tuned Classifier**
A transformer-based classifier leveraging Hugging Face’s `transformers` library.
#### Key Methods:
- **train_fold**: Fine-tunes BERT on a fold.
- **evaluate_model**: Evaluates the model on the test set.
- **save_model**: Saves model weights and tokenizer.

### 3. **Few-Shot Learning (gpt-3.5-turbo)**
Uses OpenAI’s gpt-3.5-turbo with a few predefined examples instead of full dataset training.
#### Key Methods:
- **train**: Uses predefined examples.
- **predict**: Runs inference using gpt-3.5-turbo.
- **evaluate**: Measures performance.

---

## Workflow

### **1. Data Preprocessing**
```bash
python src/utils/data_preprocessing.py
```
- Loads raw dataset (`Suicide_Detection.csv`).
- Cleans and normalizes text.
- Converts labels: `"suicide" → 1`, `"non-suicide" → 0`.
- Saves processed data.

### **2. Model Training & Evaluation**
```bash
python src/main.py
```
- Runs classifiers (`logistic`, `bert`, `few_shot`).
- Performs **5-fold cross-validation**.
- Trains final models and evaluates on test set.

### **3. Visualization**
```bash
python src/utils/visualization.py
```
- Plots **ROC curves**, **confusion matrices**, and **training metrics**.
- Saves results in `/outputs/visualizations/`.

---

## Cross-Validation & Final Model Training Process

### 1. **Train-Test Split (80-20)**
- **80% of data → Training**
- **20% of data → Testing (never used in training)**

### 2. **5-Fold Cross-Validation**
- **Training Data (80%)** split into **5 folds**.
- Model is trained **5 times**:
  - **4 folds → Training**
  - **1 fold → Validation**
- Best-performing fold is selected.

### 3. **Final Training on Full 80% Data**
- Model is **retrained using the entire 80% training data**.
- No validation during this step.

### 4. **Final Evaluation on the 20% Test Set**
- Model is tested **once on the unseen 20% data**.
- **Final metrics recorded**.

✅ **Ensures fair evaluation and avoids data leakage.**

---

## Memory Optimization Fixes
### Problem: **Excessive RAM Usage**
- **Multiple DataFrame copies created** at each preprocessing step.
- **Tokenization (for BERT) and TF-IDF Vectorization** are memory-intensive.
- **Mac M1 struggles with large Pandas DataFrames & Torch tensors.**

### 🛠️ Fixes:
1. **Dataset Sampling at the Start**
```python
df = df.sample(frac=0.1, random_state=42)
```
2. **In-place DataFrame Operations**
```python
df.dropna(subset=['text'], inplace=True)
df.drop_duplicates(subset=['text'], inplace=True)
```
3. **Reduce Batch Size in BERT Training**
4. **Limit TF-IDF Max Features**
5. **Reduce API Calls in Few-Shot Learning**

✅ **Prevents crashes and improves runtime efficiency.**

---

## Conclusion
- **Logistic Regression & BERT** follow **80/20 + 5-Fold Cross-Validation**.
- **Few-Shot Learning** uses predefined examples **without dataset training**.
- **Preprocessing optimizations** prevent excessive RAM usage.
- **Visualizations & performance tracking** provide insights into classifier behavior.

🚀 **Ready to scale and deploy for real-world applications!**

