import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

# =======================
# GPU CONFIGURATION
# =======================
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

# =======================
# REPRODUCIBILITY
# =======================
np.random.seed(123)
tf.random.set_seed(123)

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
# =======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================
# ANN MODEL
# =======================
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse"
)

# =======================
# CALLBACKS
# =======================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

# =======================
# TRAIN MODEL
# =======================
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# =======================
# SAVE MODEL & SCALER
# =======================
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Save Keras Model
model_path = os.path.join(save_dir, "ann_model.h5")
model.save(model_path)
print(f"Model saved to: {model_path}")

# Save Scaler
scaler_path = os.path.join(save_dir, "ann_scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to: {scaler_path}")

# =======================
# PREDICTIONS
# =======================
y_test_pred = model.predict(X_test_scaled).flatten()
y_train_pred = model.predict(X_train_scaled).flatten()

# =======================
# METRICS (VERSION-SAFE)
# =======================
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

print("\n===== TEST PERFORMANCE =====")
print("Test MSE :", test_mse)
print("Test RMSE:", test_rmse)
print("Test R²  :", test_r2)

print("\n===== TRAIN PERFORMANCE =====")
print("Train MSE :", train_mse)
print("Train RMSE:", train_rmse)
print("Train R²  :", train_r2)

# =======================
# PLOTS: TEST DATA
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
plt.title("ANN: Actual vs Predicted (Test)")
plt.grid(True)

plt.tight_layout()
plt.savefig("plots/ANN_Test_Actual_vs_Predicted.png", dpi=300)
plt.show()

# =======================
# PLOTS: TRAIN DATA
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
plt.title("ANN: Actual vs Predicted (Train)")
plt.grid(True)

plt.tight_layout()
plt.savefig("plots/ANN_Train_Actual_vs_Predicted.png", dpi=300)
plt.show()

