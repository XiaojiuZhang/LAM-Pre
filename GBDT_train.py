import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# ==========================================
# 1. Configuration & Data Loading
# ==========================================
DATA_INPUT = 'data/training_data.xlsx'
OUTPUT_DIR = 'results/gbdt_output/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load the dataset
# Note: Ensure the data directory and file exist before running.
df = pd.read_excel(DATA_INPUT)

# ==========================================
# 2. Feature Definitions & Train-Test Split
# ==========================================
numerical_features = [
    'cell-con.', 'hemi-con.', 'lig-con.', 'bio-loading', 'tr-time', 'tr-temp',
    'amount-1', 'MW-1', 'Covalent-1', 'XLogP-1', 'Charge-1', 'Complexity-1', 'HB Donor-1', 'HB Acceptor-1', 'Rotatable-1',
    'amount-2', 'MW-2', 'Covalent-2', 'XLogP-2', 'Charge-2', 'Complexity-2', 'HB Donor-2', 'HB Acceptor-2', 'Rotatable-2',
    'enzy-amount', 'enzy-pH', 'enzy-temp', 'enzy-time', 'enzy-rpm',
]

categorical_features = [
    'biomass-AR', 'biomass-WWC', 'biomass-GV', 'biomass-AMB', 'biomass-FPB', 'biomass-HO', 
    'chemical', 'physical', 
    'phy-microwave', 'phy-extrusion', 'phy-Ultrasonication', 'phy-milling', 'phy-hydrothermal', 'phy-explosion',
    'phy-subW', 'phy-gas', 'phy-radiation', 'phy-UHP', 
    'Ctec', 'HTec', 'Zytex', 'CelluclastTM 1.5L', 'Novo188', 'other'
]

textual_features = ['biomass name', 'chemical-1', 'Canonical Smiles-1', 'chemical-2', 'Canonical Smiles-2']

print(f"Number of numerical features: {len(numerical_features)}")

X = df[numerical_features + categorical_features + textual_features]
y = df['glu-yield'].values

# Split data FIRST to prevent data leakage
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=473)

# ==========================================
# 3. Data Preprocessing (Fit on Train, Transform Train/Test)
# ==========================================

# 3.1 Numerical Features: Standardization
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_raw[numerical_features])
X_test_num = scaler.transform(X_test_raw[numerical_features])

# 3.2 Categorical Features: One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_cat = encoder.fit_transform(X_train_raw[categorical_features])
X_test_cat = encoder.transform(X_test_raw[categorical_features])

# 3.3 Textual Features: TF-IDF Encoding
X_train_text_list = []
X_test_text_list = []

for feature in textual_features:
    tfidf_vectorizer = TfidfVectorizer(max_features=25)
    # Fit only on training data
    X_train_text_list.append(tfidf_vectorizer.fit_transform(X_train_raw[feature]).toarray())
    # Transform test data based on training vocabulary
    X_test_text_list.append(tfidf_vectorizer.transform(X_test_raw[feature]).toarray())

# Merge all processed features into final arrays
X_train = np.hstack([X_train_num, X_train_cat] + X_train_text_list)
X_test = np.hstack([X_test_num, X_test_cat] + X_test_text_list)

# ==========================================
# 4. Model Training & Hyperparameter Tuning
# ==========================================
gbdt_model = GradientBoostingRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10],
}

grid_search = GridSearchCV(estimator=gbdt_model, param_grid=param_grid, cv=5, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Save Grid Search results
cv_results = pd.DataFrame(grid_search.cv_results_)
important_columns = [
    'param_n_estimators', 'param_learning_rate', 'param_max_depth', 'param_subsample',
    'param_min_samples_split', 'param_min_samples_leaf', 'mean_test_score', 'std_test_score', 'rank_test_score'
]
cv_summary = cv_results[important_columns].sort_values(by='rank_test_score')
cv_summary.to_csv(os.path.join(OUTPUT_DIR, 'GBDT_CV_Results_Detail.csv'), index=False)

print("Best parameters found:", grid_search.best_params_)

# ==========================================
# 5. Prediction & Evaluation
# ==========================================
gbdt_best = grid_search.best_estimator_

y_train_pred = gbdt_best.predict(X_train)
y_test_pred = gbdt_best.predict(X_test)

# Metrics calculation
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)
pcc_train, _ = pearsonr(y_train, y_train_pred)  

mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)
pcc_test, _ = pearsonr(y_test, y_test_pred) 

results = {
    "Train MSE": mse_train, "Train RMSE": rmse_train, "Train MAE": mae_train, "Train R²": r2_train, "Train PCC": pcc_train,
    "Test MSE": mse_test, "Test RMSE": rmse_test, "Test MAE": mae_test, "Test R²": r2_test, "Test PCC": pcc_test
}

results_df = pd.DataFrame(results, index=['GBDT (Best Params)'])
print("\nModel Evaluation:")
print(results_df.T)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'GBDT_results.csv'), index=True)

# Save predictions for comparison
train_data = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred, 'Dataset': 'Train'})
test_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred, 'Dataset': 'Test'})
combined_data = pd.concat([train_data, test_data], ignore_index=True)
combined_data.to_csv(os.path.join(OUTPUT_DIR, 'GBDT_comparison.csv'), index=False)

# ==========================================
# 6. Visualization
# ==========================================
plt.figure(figsize=(10, 6))
plt.scatter(y_train, y_train_pred, alpha=0.5, label="Train", color="blue")
plt.scatter(y_test, y_test_pred, alpha=0.5, label="Test", color="red")
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.title("Actual vs. Predicted values - GBDT (Best Params)")
plt.legend()
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="black", lw=2, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'actual_vs_predicted.png'), dpi=300)
plt.show()