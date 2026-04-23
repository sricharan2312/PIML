from sklearn.base import BaseEstimator, RegressorMixin
# from keras.wrappers.scikit_learn import KerasRegressor # Deprecated/Removed in newer TF
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# =======================
# REPRODUCIBILITY
# =======================
np.random.seed(123)
tf.random.set_seed(123)

# Check for GPU
print("Checking for GPU...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs detected for TensorFlow. Training will use CPU.")

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
y = data[target].values

# =======================
# TRAIN–TEST SPLIT
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=123
)

# =======================
# FEATURE SCALING
# =======================x
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a custom wrapper class used for Stacking
class ANN_Regressor(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"
    
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    def __init__(self, epochs=100, batch_size=32):
        self.epochs = epochs

        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y):
        input_dim = X.shape[1]
        print(f"Training ANN: {self.epochs} epochs")
        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=2)
        return self

    def predict(self, X):
        return self.model.predict(X).flatten()

# Wrap the Keras model using the custom wrapper
# Increased batch_size to 1024 to speed up training on large dataset (200k+ rows)
# Running on CPU requires larger batches to be efficient.
ann_regressor = ANN_Regressor(epochs=50, batch_size=1024)
xgb_model = XGBRegressor(
    objective="reg:squarederror",
    eval_metric="rmse",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=123,
    n_jobs=-1,
)
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=11,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=123,
    n_jobs=-1
)


# Create a list of base estimators
base_estimators = [
    ('ANN', ann_regressor),
    ('RF', rf_model),
    ('XGB', xgb_model)
]

# Create the stacking ensemble
# Reduced cv to 3 to check progress faster (Trains 3 folds + 1 final = 4 times per model)
stacking_model = StackingRegressor(
    estimators=base_estimators,
    final_estimator=RidgeCV(),
    cv=3,
    verbose=3
)

print("Starting StackingRegressor training... This process uses 3-fold CV so each model will be trained 4 times.")
stacking_model.fit(X_train_scaled, y_train)
stacking_predictions = stacking_model.predict(X_test_scaled)

# Calculate MSE first, then take sqrt for RMSE (compatible with all sklearn versions)
stacking_mse = mean_squared_error(y_test, stacking_predictions)
stacking_rmse = np.sqrt(stacking_mse)
stacking_r2 = r2_score(y_test, stacking_predictions)

print("Stacking Test RMSE:", stacking_rmse)
print("Stacking Test MSE:", stacking_mse)
print("Stacking Test R-squared:", stacking_r2)

stacking_train_predictions = stacking_model.predict(X_train_scaled)

stacking_train_mse = mean_squared_error(y_train, stacking_train_predictions)
stacking_train_rmse = np.sqrt(stacking_train_mse)
stacking_train_r2 = r2_score(y_train, stacking_train_predictions)

print("Stacking Train RMSE:", stacking_train_rmse)
print("Stacking Train R-squared:", stacking_train_r2)

# =======================
# SAVE MODEL
# =======================
os.makedirs("saved_models", exist_ok=True)
joblib.dump(stacking_model, "saved_models/stacking_model.pkl")
print("Model saved to saved_models/stacking_model.pkl")




