import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

# =======================
# CREATE PLOT DIRECTORY
# =======================
os.makedirs("plots", exist_ok=True)

# =======================
# DATA LOADING
# =======================
data = pd.read_excel("Dataset/Training Dataset/CV_DATASET (2).xlsx")

predictors = ["Potential", "OXIDATION", "Zn/Co_Conc", "SCAN_RATE", "ZN", "CO"]
target = "Current"

X = data[predictors]
y = data[target]

# =======================
# TRAIN–TEST SPLIT
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=123
)

# =======================
# RANDOM FOREST MODEL
# =======================
# Note: RandomForestRegressor in scikit-learn does not support GPU acceleration.
# It uses CPU parallel processing via n_jobs=-1.
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=11,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=123,
    n_jobs=-1,
    verbose=2
)

# =======================
# TRAIN MODEL
# =======================
rf_model.fit(X_train, y_train)

# =======================
# PREDICTIONS
# =======================
y_test_pred = rf_model.predict(X_test)
y_train_pred = rf_model.predict(X_train)

# =======================
# METRICS
# =======================
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

print("===== TEST PERFORMANCE =====")
print("Test MSE :", test_mse)
print("Test RMSE:", test_rmse)
print("Test R²  :", test_r2)

print("\n===== TRAIN PERFORMANCE =====")
print("Train MSE :", train_mse)
print("Train RMSE:", train_rmse)
print("Train R²  :", train_r2)

# =======================
# FEATURE IMPORTANCE
# =======================
feature_importance = (
    pd.Series(rf_model.feature_importances_, index=predictors)
    .sort_values(ascending=False)
)

print("\nFeature Importance:")
print(feature_importance)

# =======================
# FEATURE IMPORTANCE PLOT
# =======================
plt.figure(figsize=(7, 4))
feature_importance.plot(kind="bar")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.grid(axis="y")

plt.tight_layout()
plt.savefig("plots/RF_Feature_Importance.png", dpi=300)
plt.show()

# =======================
# ACTUAL vs PREDICTED — TEST
# =======================
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "k--", lw=2
)
plt.xlabel("Actual Current")
plt.ylabel("Predicted Current")
plt.title("RF: Actual vs Predicted (Test)")
plt.grid(True)

plt.tight_layout()
plt.savefig("plots/RF_Test_Actual_vs_Predicted.png", dpi=300)
plt.show()

# =======================
# ACTUAL vs PREDICTED — TRAIN
# =======================
plt.figure(figsize=(6, 6))
plt.scatter(y_train, y_train_pred, alpha=0.7, color="red")
plt.plot(
    [y_train.min(), y_train.max()],
    [y_train.min(), y_train.max()],
    "k--", lw=2
)
plt.xlabel("Actual Current")
plt.ylabel("Predicted Current")
plt.title("RF: Actual vs Predicted (Train)")
plt.grid(True)

plt.tight_layout()
plt.savefig("plots/RF_Train_Actual_vs_Predicted.png", dpi=300)
plt.show()

# =======================
# 10-FOLD CROSS-VALIDATION
# =======================
cv = KFold(n_splits=10, shuffle=True, random_state=123)

cv_r2_scores = cross_val_score(
    rf_model,
    X, y,
    scoring="r2",
    cv=cv,
    n_jobs=-1
)

print("\n10-Fold Cross-Validation R² Scores:")
for i, score in enumerate(cv_r2_scores, 1):
    print(f"Fold {i}: {score:.4f}")

# =======================
# SAVE MODEL
# =======================
os.makedirs("saved_models", exist_ok=True)
joblib.dump(rf_model, "saved_models/rf_model.pkl")
print("Model saved to saved_models/rf_model.pkl")

print("\nAverage CV R²:", np.mean(cv_r2_scores))
print("Std CV R²     :", np.std(cv_r2_scores))
