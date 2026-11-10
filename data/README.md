# Datasets

This directory contains or provides access to the datasets used in the **Anomaly Detection & Machine Learning Benchmark** project.

The project covers three use cases: unsupervised anomaly detection, credit card fraud detection, and predictive maintenance.

## 1. Mouse Dataset

**File:** `mouse.txt`

- **Observations:** 500
- **Features:** 2 numerical variables (`x`, `y`)
- **Normal observations:** 490
- **Anomalies:** 10

The Mouse dataset is used to compare Isolation Forest and Local Outlier Factor (LOF), investigate anomaly detection thresholds, and evaluate novelty detection and robustness to training contamination.

The dataset is stored locally in this directory.

**Source:** Original dataset provided as part of the practical assignment. Redistribution permissions should be verified before publication.

## 2. Credit Card Fraud Detection

**File:** `creditcard.csv`

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- **Observations:** 284,807
- **Features:** Anonymized transaction features, transaction time, and amount
- **Target:** `Class` (0 = legitimate transaction, 1 = fraud)

The dataset is used for supervised classification, unsupervised anomaly detection, threshold optimization, and model explainability.

Due to its size, the CSV file is excluded from Git tracking.

The corresponding notebook automatically downloads the dataset using `kagglehub` if it is not available locally.

## 3. AI4I 2020 Predictive Maintenance

**File:** `ai4i2020.csv`

**Sources:**
- [Kaggle — AI4I 2020](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
- [UCI Machine Learning Repository — AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i)

- **Observations:** 10,000
- **Features:** Machine characteristics and operating measurements
- **Target:** `Machine failure` (0 = normal operation, 1 = failure)

This synthetic dataset is used to evaluate anomaly detection and supervised classification methods in a predictive maintenance setting.

The corresponding notebook automatically downloads the dataset through `kagglehub` when the local CSV file is unavailable.

## Data Access

The notebooks use the following strategy:

1. Check whether the required dataset exists in the local `data/` directory.
2. If unavailable, download the Kaggle dataset using `kagglehub`.
3. Load the dataset into Pandas for analysis.

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

An Internet connection may be required for the initial download. Kaggle authentication may also be necessary depending on dataset access requirements.

## Notes

- All datasets are used for educational and research purposes.
- The original datasets are not modified.
- Credit Card and AI4I datasets are excluded from Git tracking and downloaded when necessary.
- The AI4I dataset contains synthetic observations and does not represent actual industrial failure records.
- Dataset ownership and redistribution rights remain with their respective providers.
