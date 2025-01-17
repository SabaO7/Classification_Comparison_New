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

### 2. **Train Classifiers**

- Train the BERT classifier:

```bash
python classifiers/bert_classifier.py
```

- Train the logistic regression classifier:

```bash
python classifiers/logistic_classifier.py
```

- Train the few-shot learning classifier:

```bash
python classifiers/few_shot_classifier.py
```

### 3. **Evaluate Models**

- Use evaluation methods provided in each classifier to compute metrics on the test set.

### 4. **Visualize Metrics**

- Run `visualization.py` to generate performance plots.

---

## Debugging

### 1. **Dataset Inspection**

Use `debug_bert.py` to debug tokenized datasets:

```bash
python classifiers/debug_bert.py
```

### 2. **Batch Inspection**

Inspect input shapes and data types during training:

```python
for batch in train_loader:
    print(f"Batch Input IDs: {batch['input_ids'].shape}")
    print(f"Batch Attention Mask: {batch['attention_mask'].shape}")
    print(f"Batch Labels: {batch['labels'].shape}")
    break
```

---

## Outputs

- **Models**: Saved in `outputs/models/` with timestamps.
- **Logs**: Training and evaluation logs in `outputs/logs/`.
- **Metrics**: JSON files with metrics for analysis.

---

## NEXT STEP
- debug the bert error Error: np.int64(7) - it happened after I did this "Limit Dataset loading in memory, currently, your TextDataset class holds the entire dataset in memory.  For large datasets, use a dataloader with a custom _getitem_ to load data dynamically"

