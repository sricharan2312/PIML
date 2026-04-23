"""
PIML Training Pipeline
=======================
End-to-end training script for the Physics-Informed ML model.
Handles data loading, preprocessing, model construction, training
with physics constraints, and model persistence.

Usage:
    python PIML_training.py                    # train with synthetic data
    python PIML_training.py --use-experimental # train with real Excel data
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Local imports
from dataset_generator import generate_synthetic_dataset, load_experimental_data, augment_dataset
from PIML_model import (
    build_piml_ann,
    build_piml_lstm,
    build_piml_transformer,
    build_piml_multioutput,
    PhysicsLoss,
    PIMLTrainer,
)

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Paths
    "model_dir":      "saved_models",
    "plot_dir":       "plots",
    "data_dir":       "Dataset",

    # Features
    "feature_cols": [
        "Potential", "OXIDATION", "Zn_Co_Conc", "Scan_Rate",
        "ZN", "CO", "Temperature", "Electrode_Area",
    ],
    "target_col":     "Current_Density",

    # Architecture
    "architecture":   "ann",          # "ann", "lstm", "transformer"
    "hidden_units":   [256, 128, 64, 32],
    "dropout_rate":   0.05,
    "mc_dropout":     True,

    # Physics loss weights – 9 electrochemical constraints
    # (kept small – physics terms are much larger in magnitude than MSE)
    "lambda_faraday":              0.001,
    "lambda_capacitance":          0.001,
    "lambda_redox":                0.001,
    "lambda_smooth":               0.001,
    "lambda_butler_volmer":        0.0005,  # Butler-Volmer kinetics
    "lambda_nernst":               0.0005,  # Nernst temperature dependence
    "lambda_charge_conservation":  0.001,   # ∮I·dV ≈ 0
    "lambda_randles_sevcik":       0.0005,  # I_peak ∝ √v
    "lambda_thermodynamic":        0.0002,  # DeltaG = -nFE consistency
    "max_current":                 5000.0,  # mA/cm² – realistic upper bound

    # Training
    "epochs":         500,
    "batch_size":     256,
    "learning_rate":  5e-3,
    "min_lr":         1e-6,       # minimum LR for cosine annealing
    "patience":       50,
    "warmup_epochs":  80,
    "test_size":      0.15,
    "val_size":       0.15,
    "seed":           42,

    # Uncertainty
    "mc_samples":     50,

    # Augmentation
    "augment":        True,

    # Cosine annealing with warm restarts
    "use_cosine_annealing": True,
    "cosine_T_0":     100,       # first restart cycle length
    "cosine_T_mult":  2,         # cycle length multiplier
}


def setup_gpu():
    """Configure GPU memory growth."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        print(f"GPU(s) available: {len(gpus)}")
    else:
        print("No GPU detected – using CPU.")


def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(cfg: dict, use_experimental: bool = False):
    """Load, validate, split, and scale data."""

    if use_experimental:
        print("Loading experimental dataset...")
        df = load_experimental_data()
    else:
        csv_path = os.path.join(cfg["data_dir"], "synthetic_cv_dataset.csv")
        if os.path.exists(csv_path):
            print(f"Loading existing synthetic dataset from {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            print("Generating synthetic dataset...")
            df = generate_synthetic_dataset(
                n_potential_points=200,
                noise_std=0.03,
                save_path=csv_path,
            )

    # --- Data Augmentation ---
    if cfg.get("augment", True):
        print("Applying data augmentation...")
        df = augment_dataset(
            df, noise_factor=0.02, n_augmented=1,
            interpolation=True, seed=cfg["seed"],
        )

    # Ensure all required columns exist
    available = set(df.columns)
    feature_cols = [c for c in cfg["feature_cols"] if c in available]
    if cfg["target_col"] not in available:
        raise KeyError(f"Target column '{cfg['target_col']}' not in dataset. "
                        f"Available: {sorted(available)}")

    print(f"Features used ({len(feature_cols)}): {feature_cols}")
    print(f"Target: {cfg['target_col']}")
    print(f"Dataset shape: {df.shape}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[cfg["target_col"]].values.astype(np.float32)

    # --- Three-way split: train / val / test ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["seed"]
    )
    relative_val = cfg["val_size"] / (1.0 - cfg["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, random_state=cfg["seed"]
    )

    print(f"Splits  ->  train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # --- Scale features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled   = scaler.transform(X_val).astype(np.float32)
    X_test_scaled  = scaler.transform(X_test).astype(np.float32)

    # --- Scale target variable ---
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
    y_val_scaled   = y_scaler.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)
    y_test_scaled  = y_scaler.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)
    print(f"Target stats  ->  mean={y_scaler.mean_[0]:.4f}, std={y_scaler.scale_[0]:.4f}")

    # Save scalers
    os.makedirs(cfg["model_dir"], exist_ok=True)
    scaler_path = os.path.join(cfg["model_dir"], "piml_scaler.pkl")
    y_scaler_path = os.path.join(cfg["model_dir"], "piml_y_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    joblib.dump(y_scaler, y_scaler_path)
    print(f"Scalers saved -> {scaler_path}, {y_scaler_path}")

    return {
        "X_train_scaled": X_train_scaled, "X_train_raw": X_train,
        "X_val_scaled":   X_val_scaled,   "X_val_raw":   X_val,
        "X_test_scaled":  X_test_scaled,  "X_test_raw":  X_test,
        "y_train": y_train_scaled, "y_val": y_val_scaled, "y_test": y_test_scaled,
        "y_train_raw": y_train, "y_val_raw": y_val, "y_test_raw": y_test,
        "scaler": scaler,
        "y_scaler": y_scaler,
        "feature_cols": feature_cols,
        "df": df,
    }


# ============================================================
# BUILD + TRAIN
# ============================================================

def build_model(cfg: dict, input_dim: int):
    """Construct the PIML model based on config."""
    arch = cfg["architecture"].lower()

    if arch == "ann":
        model = build_piml_ann(
            input_dim=input_dim,
            hidden_units=cfg["hidden_units"],
            dropout_rate=cfg["dropout_rate"],
            mc_dropout=cfg["mc_dropout"],
        )
    elif arch == "lstm":
        # LSTM expects (batch, seq_len, features)
        # For point-wise data we reshape: seq_len=1
        model = build_piml_lstm(seq_len=1, n_features=input_dim)
    elif arch == "transformer":
        model = build_piml_transformer(seq_len=1, n_features=input_dim)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model.summary()
    return model


def train_piml(cfg: dict, data: dict):
    """Train the PIML model with physics-informed loss."""

    model = build_model(cfg, input_dim=data["X_train_scaled"].shape[1])

    physics_loss = PhysicsLoss(
        lambda_faraday=cfg["lambda_faraday"],
        lambda_capacitance=cfg["lambda_capacitance"],
        lambda_redox=cfg["lambda_redox"],
        lambda_smooth=cfg["lambda_smooth"],
        lambda_butler_volmer=cfg.get("lambda_butler_volmer", 0.0005),
        lambda_nernst=cfg.get("lambda_nernst", 0.0005),
        lambda_charge_conservation=cfg.get("lambda_charge_conservation", 0.001),
        lambda_randles_sevcik=cfg.get("lambda_randles_sevcik", 0.0005),
        lambda_thermodynamic=cfg.get("lambda_thermodynamic", 0.0002),
        max_current_density=cfg["max_current"],
        y_mean=float(data["y_scaler"].mean_[0]),
        y_scale=float(data["y_scaler"].scale_[0]),
    )

    # Cosine annealing with warm restarts for learning rate
    if cfg.get("use_cosine_annealing", False):
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=cfg["learning_rate"],
            first_decay_steps=cfg.get("cosine_T_0", 100),
            t_mul=cfg.get("cosine_T_mult", 2),
            alpha=cfg.get("min_lr", 1e-6) / cfg["learning_rate"],
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg["learning_rate"])

    trainer = PIMLTrainer(
        model=model,
        physics_loss=physics_loss,
        optimizer=optimizer,
        feature_columns=data["feature_cols"],
    )

    print("\n" + "=" * 60)
    print("  PIML TRAINING  –  Physics-Informed Hybrid Loss (9 Constraints)")
    print("  *** ReLoBRaLo Adaptive Weighting ENABLED ***")
    print("=" * 60)
    print(f"  lambda_faraday              = {cfg['lambda_faraday']}")
    print(f"  lambda_capacitance          = {cfg['lambda_capacitance']}")
    print(f"  lambda_redox                = {cfg['lambda_redox']}")
    print(f"  lambda_smooth               = {cfg['lambda_smooth']}")
    print(f"  lambda_butler_volmer        = {cfg.get('lambda_butler_volmer', 0.0005)}")
    print(f"  lambda_nernst               = {cfg.get('lambda_nernst', 0.0005)}")
    print(f"  lambda_charge_conservation  = {cfg.get('lambda_charge_conservation', 0.001)}")
    print(f"  lambda_randles_sevcik       = {cfg.get('lambda_randles_sevcik', 0.0005)}")
    print(f"  lambda_thermodynamic        = {cfg.get('lambda_thermodynamic', 0.0002)}")
    print(f"  warmup_epochs          = {cfg['warmup_epochs']}")
    print(f"  cosine_annealing       = {cfg.get('use_cosine_annealing', False)}")
    print("=" * 60 + "\n")

    history = trainer.fit(
        X_train_scaled=data["X_train_scaled"],
        X_train_raw=data["X_train_raw"],
        y_train=data["y_train"],
        X_val_scaled=data["X_val_scaled"],
        X_val_raw=data["X_val_raw"],
        y_val=data["y_val"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        patience=cfg["patience"],
        warmup_epochs=cfg["warmup_epochs"],
        verbose=1,
    )

    # --- Save model ---
    os.makedirs(cfg["model_dir"], exist_ok=True)
    model_path = os.path.join(cfg["model_dir"], "piml_model.keras")
    model.save(model_path)
    print(f"Model saved -> {model_path}")

    # Save training history
    hist_path = os.path.join(cfg["model_dir"], "piml_history.json")
    serialisable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(hist_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"History saved -> {hist_path}")

    # Save config
    cfg_path = os.path.join(cfg["model_dir"], "piml_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    return trainer, history


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train PIML model")
    parser.add_argument("--use-experimental", action="store_true",
                        help="Use real experimental Excel data instead of synthetic")
    parser.add_argument("--arch", type=str, default="ann",
                        choices=["ann", "lstm", "transformer"],
                        help="Model architecture")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda-faraday", type=float, default=None)
    parser.add_argument("--lambda-redox", type=float, default=None)
    parser.add_argument("--lambda-bv", type=float, default=None,
                        help="Butler-Volmer kinetics constraint weight")
    parser.add_argument("--lambda-nernst", type=float, default=None,
                        help="Nernst temperature constraint weight")
    parser.add_argument("--lambda-charge", type=float, default=None,
                        help="Charge conservation constraint weight")
    parser.add_argument("--lambda-randles", type=float, default=None,
                        help="Randles-Sevcik constraint weight")
    parser.add_argument("--no-cosine", action="store_true",
                        help="Disable cosine annealing LR schedule")
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg["architecture"] = args.arch
    if args.epochs:     cfg["epochs"] = args.epochs
    if args.lr:         cfg["learning_rate"] = args.lr
    if args.lambda_faraday is not None: cfg["lambda_faraday"] = args.lambda_faraday
    if args.lambda_redox is not None:   cfg["lambda_redox"] = args.lambda_redox
    if args.lambda_bv is not None:      cfg["lambda_butler_volmer"] = args.lambda_bv
    if args.lambda_nernst is not None:  cfg["lambda_nernst"] = args.lambda_nernst
    if args.lambda_charge is not None:  cfg["lambda_charge_conservation"] = args.lambda_charge
    if args.lambda_randles is not None: cfg["lambda_randles_sevcik"] = args.lambda_randles
    if args.no_cosine:                  cfg["use_cosine_annealing"] = False

    setup_gpu()
    set_seeds(cfg["seed"])

    # 1. Prepare data
    data = prepare_data(cfg, use_experimental=args.use_experimental)

    # 2. Train
    trainer, history = train_piml(cfg, data)

    # 3. Quick test evaluation
    print("\n===== TEST SET EVALUATION =====")
    from PIML_evaluation import evaluate_piml, extrapolation_analysis
    evaluate_piml(trainer, data, cfg)

    # 4. Extrapolation analysis
    print("\n===== EXTRAPOLATION ANALYSIS =====")
    extrap_results = extrapolation_analysis(
        trainer.model, data["scaler"], data["feature_cols"],
        save_path=os.path.join(cfg["plot_dir"], "PIML_extrapolation.png"),
    )
    print(f"Randles-Sevcik R²: {extrap_results.get('randles_sevcik_r2', 0):.4f}")

    print("\nTraining pipeline complete.")


if __name__ == "__main__":
    main()
