# Wearable Physiological Stress Classification & Multi-Horizon Forecasting Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](#-getting-started)

An end-to-end deep learning and machine learning pipeline for **3-class current stress classification** (No Stress, Low Stress, High Stress) and **exploratory multi-horizon stress state forecasting** (30s, 60s, 120s, 300s) using multi-modal wearable sensor streams (`HR`, `EDA`, `TEMP`, 3-axis `Accelerometer`).

Designed with strict data integrity standards: zero-data-leakage participant-disjoint splits, actual-timestamp event matching, and validation-first model selection.

---

## 📌 Key Highlights

* **Leakage-Free Validation:** Participant splits are strictly frozen upfront (9 Train / 3 Val / 3 Test). Imputers and scalers fit exclusively on training data to prevent lookahead bias.
* **Timestamp-Based Horizon Matching:** Target state mapping uses actual recorded timestamps within strict per-horizon tolerance windows ($\pm 1.0\text{s}$ to $\pm 5.0\text{s}$) rather than assumed fixed sampling rates.
* **Session & Gap Safety Audits:** Enforces temporal continuity across $11.5\text{M+}$ raw rows and $105\text{k+}$ inferred sessions, discarding windows spanning signal dropouts $>5.0\text{s}$.
* **Baseline Benchmarking:** Compares deep sequence models (LSTM, GRU) against Majority baselines, Logistic Regression (SAGA), and Random Forests.

---

## 📊 Dataset Overview

* **Raw Observations:** 11,509,051 rows across 15 participants
* **Sensor Features:** Heart Rate (`HR`), Electrodermal Activity (`EDA`), Skin Temperature (`TEMP`), 3-axis Acceleration (`X`, `Y`, `Z`)
* **Window Configuration:** Sequence length of 32 steps with a stride of 8 steps

| Class | Label | Overall Row Count | Overall Percentage |
| :--- | :---: | :---: | :---: |
| **No Stress** | `0` | 2,162,246 | 18.79% |
| **Low Stress** | `1` | 806,222 | 7.01% |
| **High Stress** | `2` | 8,540,583 | 74.21% |

### Frozen Participant Partitioning

* **Train (`9` subjects):** `15`, `6B`, `6D`, `8B`, `BG`, `DF`, `E4`, `EG`, `F5` (761,291 windows)
* **Validation (`3` subjects):** `7A`, `7E`, `CE` (297,308 windows)
* **Test (`3` subjects):** `5C`, `83`, `94` (352,965 windows)

---

## 📈 Experimental Results

### 1. Three-Class Current Stress Classification (Test Set)

| Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | Macro OvR AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.5676 | 0.3333 | 0.1892 | 0.3333 | 0.2414 | 0.4110 | N/A |
| **Logistic Regression** | 0.5439 | 0.4110 | 0.4341 | 0.4110 | **0.4117** | 0.5249 | 0.5714 |
| **Random Forest** | **0.6051** | **0.4150** | 0.3821 | **0.4150** | 0.3890 | **0.5455** | 0.5698 |
| **GRU** | 0.5411 | 0.3573 | 0.3402 | 0.3573 | 0.3313 | 0.4811 | 0.6068 |
| **LSTM** | 0.5645 | 0.3786 | 0.3663 | 0.3786 | 0.3539 | 0.5072 | **0.6166** |

---

### 2. Multi-Horizon Forecasting Verification

| Horizon | Requested ($\Delta t$) | Window Count | Actual Mean $\Delta t$ | Actual Std $\Delta t$ | Match Tolerance |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **30s** | 30s | 347,745 | 30.000s | 0.015s | $\pm 1.0\text{s}$ |
| **60s** | 60s | 342,345 | 60.000s | 0.015s | $\pm 1.0\text{s}$ |
| **120s** | 120s | 331,725 | 119.999s | 0.041s | $\pm 2.0\text{s}$ |
| **300s** | 300s | 300,551 | 299.993s | 0.159s | $\pm 5.0\text{s}$ |

---

## ⚙️ Getting Started

### Google Colab Execution (Recommended)

1. Upload your dataset (`Paper_nurse_dataset.zip`) to Google Drive.
2. Open the `.ipynb` notebook directly in Google Colab.
3. Ensure runtime is set to GPU (**Runtime > Change runtime type > T4 GPU**).
4. Run all cells sequentially.

### Dependencies

To execute locally, install the following core packages:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
