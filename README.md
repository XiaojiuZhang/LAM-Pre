# LAM-Pre: A Generalizable ML Framework for Biorefinery Pretreatment Design

![LAM-Pre](./image/LAM-Pre.png)

## Overview
LAM-Pre integrates Large Language Model (LLM)-assisted literature mining, systematic Machine Learning (ML) benchmarking, and experimental validation to predict glucose yields across diverse biomass feedstocks and pretreatment conditions.

By extracting data from over 14 years of literature, we established the Pret-Glucose dataset, comprising 305 biomass types and 1,473 pretreatment strategies. This framework serves as a scalable, data-driven paradigm for flexible and economically viable biorefinery design.


## 📌 Key Features
* **LLM-Assisted Extraction**: Automated pipeline using ChatGPT-4 for structuring fragmented literature data.
* **Robust Benchmarking**: Comparison of 16 supervised models, identifying Gradient Boosting Decision Trees (GBDT) as the optimal predictor ($R^2 = 0.672$).
* **Explainable AI**: SHAP and PDP analysis to reveal non-linear interactions between enzyme loading, temperature, and biomass composition.
* **Experimental Validation**: Demonstrated PCC $> 0.9$ in diverse feedstock categories and successfully identified high-yield (96%) process regimes.

## 📂 Directory Structure

LAM-Pre/
├── model_pkl/              # Pre-trained models and preprocessors 
│   ├── trained_gbdt_model.pkl    # Optimized GBDT model weights
│   ├── scaler.pkl                # Numerical feature standardizer
│   ├── encoder.pkl               # Categorical feature encoder
│   └── tfidf_vectorizers.pkl     # TF-IDF vectorizers for textual data
├── validation/             # Scripts for model validation and generalization
│   └── model_validation.py       # Independent dataset testing
├── image/                  # Figures and logos for documentation
├── GBDT_train.py           # Core script for training and hyperparameter tuning
├── batch_predict.py        # Script for batch glucose yield prediction
├── batch_predict_mixed_biomass.py # Prediction script for biomass blending strategies [cite: 114]
├── environment.yml         # Conda environment configuration
└── LICENSE                 # MIT License

## 🚀 Quick Start

### 1. Installation
To set up the environment and install necessary dependencies, run the following commands in your terminal:
```bash
# Clone the repository
git clone https://github.com/XiaojiuZhang/LAM-Pre.git
cd LAM-Pre
# Create and activate the environment
conda env create -f environment.yml
conda activate lampre
```

### 2. Model Training
To retrain the GBDT model using the Pret-Glucose dataset:
```bash
python GBDT_train.py
```

### 3. Batch Prediction
To predict glucose yields for new process conditions using pre-trained weights:
```bash
# For single feedstock scenarios
python batch_predict.py

# For mixed biomass blending strategies
python batch_predict_mixed_biomass.py
```

### 4. Generalization Validation
To test the model on the independent benchmark dataset (2024-2026 literature):
```bash
python validation/model_validation.py
```
