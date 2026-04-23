"""Quick diagnostic to trace origin of NaN in training."""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIML_model import build_piml_ann, PhysicsLoss

print("=== DATA CHECK ===")
df = pd.read_csv("Dataset/synthetic_cv_dataset.csv")
print(f"Shape: {df.shape}, NaN: {df.isna().sum().sum()}, "
      f"Inf: {np.isinf(df.select_dtypes(include=[np.number]).values).sum()}")

feat = ["Potential","OXIDATION","Zn_Co_Conc","Scan_Rate","ZN","CO","Temperature","Electrode_Area"]
X = df[feat].values.astype(np.float32)
y = df["Current_Density"].values.astype(np.float32)
print(f"y: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}, std={y.std():.2f}")
print(f"y NaN: {np.isnan(y).sum()}, Inf: {np.isinf(y).sum()}")

sx = StandardScaler(); Xsc = sx.fit_transform(X).astype(np.float32)
sy = StandardScaler(); ysc = sy.fit_transform(y.reshape(-1,1)).flatten().astype(np.float32)
print(f"Xsc NaN: {np.isnan(Xsc).sum()}, ysc NaN: {np.isnan(ysc).sum()}")

print("\n=== MODEL CHECK ===")
model = build_piml_ann(8)
Xb = tf.constant(Xsc[:256])
Xr = tf.constant(X[:256])
yb = tf.constant(ysc[:256])
pred = model(Xb, training=True)
print(f"pred: NaN={bool(tf.reduce_any(tf.math.is_nan(pred)))}, "
      f"min={float(tf.reduce_min(pred)):.4f}, max={float(tf.reduce_max(pred)):.4f}")

print("\n=== INDIVIDUAL LOSSES (eager) ===")
pl = PhysicsLoss(
    max_current_density=5000.0,
    y_mean=float(sy.mean_[0]),
    y_scale=float(sy.scale_[0]),
)
for name, fn in [
    ("MSE",      lambda: pl.mse_loss(yb, pred)),
    ("Faraday",  lambda: pl.faraday_loss(pred, Xr)),
    ("Capacit",  lambda: pl.capacitance_loss(pred, Xr)),
    ("Redox",    lambda: pl.redox_limit_loss(pred)),
    ("Smooth",   lambda: pl.smoothness_loss(pred)),
    ("BV",       lambda: pl.butler_volmer_loss(pred, Xr)),
    ("Nernst",   lambda: pl.nernst_loss(pred, Xr)),
    ("Charge",   lambda: pl.charge_conservation_loss(pred, Xr)),
    ("Randles",  lambda: pl.randles_sevcik_loss(pred, Xr)),
    ("Thermo",   lambda: pl.thermodynamic_loss(pred, Xr)),
]:
    val = fn()
    print(f"  {name:10s}: {float(val):12.4f}  NaN={bool(tf.math.is_nan(val))}")

print("\n=== TOTAL LOSS (eager, pw=0) ===")
total, comps = pl.total_loss(yb, pred, Xr, physics_weight=0.0)
print(f"total = {float(total):.4f}, NaN = {bool(tf.math.is_nan(total))}")

print("\n=== GRADIENT CHECK (eager, pw=0) ===")
with tf.GradientTape() as tape:
    p2 = model(Xb, training=True)
    loss, _ = pl.total_loss(yb, p2, Xr, physics_weight=0.0)
grads = tape.gradient(loss, model.trainable_variables)
nan_count = sum(1 for g in grads if g is not None and tf.reduce_any(tf.math.is_nan(g)).numpy())
none_count = sum(1 for g in grads if g is None)
print(f"Gradients: total={len(grads)}, None={none_count}, NaN={nan_count}")
if nan_count > 0:
    for i, g in enumerate(grads):
        if g is not None and tf.reduce_any(tf.math.is_nan(g)).numpy():
            print(f"  Var {model.trainable_variables[i].name}: NaN grad")

print("\n=== GRADIENT CHECK (eager, pw=1) ===")
with tf.GradientTape() as tape:
    p3 = model(Xb, training=True)
    loss2, _ = pl.total_loss(yb, p3, Xr, physics_weight=1.0)
grads2 = tape.gradient(loss2, model.trainable_variables)
nan_count2 = sum(1 for g in grads2 if g is not None and tf.reduce_any(tf.math.is_nan(g)).numpy())
print(f"loss(pw=1) = {float(loss2):.4f}, NaN = {bool(tf.math.is_nan(loss2))}")
print(f"Gradients: NaN={nan_count2}")

print("\n=== @tf.function GRADIENT CHECK (pw=0) ===")
@tf.function
def train_step(model, X_s, X_r, y, pw):
    with tf.GradientTape() as tape:
        y_pred = model(X_s, training=True)
        loss, comps = pl.total_loss(y, y_pred, X_r, pw)
    grads = tape.gradient(loss, model.trainable_variables)
    return loss, grads

model2 = build_piml_ann(8)  # fresh model
loss_tf, grads_tf = train_step(model2, Xb, Xr, yb, tf.constant(0.0))
nan_g = sum(1 for g in grads_tf if g is not None and tf.reduce_any(tf.math.is_nan(g)).numpy())
print(f"loss = {float(loss_tf):.4f}, NaN = {bool(tf.math.is_nan(loss_tf))}")
print(f"Gradients: NaN={nan_g}")

print("\nDiagnostic complete.")
