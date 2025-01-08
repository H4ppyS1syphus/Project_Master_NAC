# Pulse Shape Discrimination - README

## 1. Overview

This repository provides tools and code for **pulse shape discrimination** of scintillation signals, focusing on distinguishing **Li6** and **Po** events. It includes:

- **Jupyter Notebook**: Demonstrates dataset exploration, feature engineering, and model-building steps (both classical ML and deep learning).
- **Utilities** (`utils/` directory): Reusable modules for data loading, feature extraction, model training, and uncertainty estimation.
- **Data** (`data/` directory): Sample datasets (`.npz` files) used for illustration and testing, including `Li6.npz`, `Po.npz`, and `Phys.npz`.

## 2. Environment Setup

To ensure reproducibility, follow these steps to create a Python virtual environment and install the required packages.

### 2.1. Python Environment Creation

- **Windows**:

  ```bash
  python -m venv venv
  venv\\Scripts\\activate
  ```

- **macOS (or Linux)**:

  ```bash
  python3 -m venv venv 
  source venv/bin/activate
  ```

### 2.2. Install Required Packages

In either Windows or macOS, with your virtual environment active, install all dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## 3. Repository Contents

├── data/
│   ├── Li6.npz
│   ├── Po.npz
│   └── Phys.npz
├── utils/
│   ├── data_loading.py
│   ├── features_extraction.py
│   ├── ml_classifiers.py
│   ├── deep_learning.py
│   ├── waveforms_analysis.py
│   ├── features_visualization.py
│   ├── uncertainty.py
│   └── analysis.py
├── PulseShape_Discrimination.ipynb
├── requirements.txt
└── README.md

### 3.1. PulseShape_Discrimination.ipynb
A Jupyter Notebook presenting the main workflow:

- Dataset Exploration: Loading Li6 and Po signals, visualizing waveforms and basic statistics.
- Feature Extraction: Computing time-domain and frequency-domain features for each detector channel.
- Machine Learning Classifiers: Training and evaluation of RandomForest, SVM, and XGBoost models.
- Deep Learning Approaches: Training CNN architectures (single-branch, multi-branch, and attention-based).
- Uncertainty Estimation: Applying Monte Carlo Dropout to quantify epistemic uncertainty.
- Po Contamination Control: Adjusting classification thresholds to cap Po contamination at 5%.
- Phys Dataset Classification: Classifying signals from the unlabeled Phys dataset.

### 3.2. utils/ Directory
Contains reusable modules and helper functions:

- data_loading.py: Functions to load .npz files and prepare waveforms.
- features_extraction.py: Utilities for computing time-domain and frequency-domain features.
- ml_classifiers.py: Simple ML training routines (RandomForest, SVM, XGBoost).
- deep_learning.py: PyTorch-based CNN model definitions and training loops.
- waveforms_analysis.py: Plotting routines for waveform statistics and comparisons.
- features_visualization.py: Methods to visualize extracted features (histograms, pair plots, etc.).
- uncertainty.py: Methods to enable dropout at inference and compute predictive distributions for epistemic uncertainty.
- analysis.py: Functions for threshold-based contamination control and classification post-processing.

### 3.3. data/ Directory
- Li6.npz, Po.npz: Example scintillation signal datasets used for demonstration.
- Phys.npz: Unlabeled real-world dataset on which the final model inference is performed.

## 4. Usage Instructions

1. Clone this repository (or download the ZIP) to your local machine.

2. Activate the Python environment (see Section 2).

3. Install dependencies:

 ```bash
  pip install -r requirements.txt
  ```

5. Run the notebook

