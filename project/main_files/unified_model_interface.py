import os
import json
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Sklearn & XGBoost
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor

# PIML imports
from PIML_model import build_piml_ann, PhysicsLoss, PIMLTrainer
from PIML_evaluation import (
    evaluate_piml, compute_metrics, compute_constraint_violations,
    plot_pred_vs_actual, plot_cv_curves, plot_training_history,
    plot_shap_analysis, analyse_optimal_composition,
    plot_uncertainty, plot_redox_peaks,
    extrapolation_analysis, compare_all_models, generate_latex_table,
    plot_butler_volmer_adherence, plot_physics_dashboard,
    generate_physics_report,
)
from dataset_generator import generate_synthetic_dataset, load_experimental_data, augment_dataset

# Keras Wrappers
try:
    from keras.wrappers.scikit_learn import KerasRegressor
except ImportError:
    pass

# =======================
# CONFIGURATION
# =======================
DATA_PATH = "Dataset/Training Dataset/CV_DATASET (2).xlsx"
MODEL_DIR = "saved_models"
PLOT_DIR = "plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# PIML default feature columns (superset of original)
PIML_FEATURE_COLS = [
    "Potential", "OXIDATION", "Zn_Co_Conc", "Scan_Rate",
    "ZN", "CO", "Temperature", "Electrode_Area",
]

# =======================
# HELPERS
# =======================
def load_data():
    if not os.path.exists(DATA_PATH):
        # Try finding it relative to project root if run from different dir
        alt_path = os.path.join("project", DATA_PATH)
        if os.path.exists(alt_path):
            return pd.read_excel(alt_path)
        raise FileNotFoundError(f"Could not find dataset at {DATA_PATH}")

    return pd.read_excel(DATA_PATH)

def preprocess_data(data):
    predictors = ["Potential", "OXIDATION", "Zn/Co_Conc", "SCAN_RATE", "ZN", "CO"]
    target = "Current"
    
    X = data[predictors]
    y = data[target].values
    
    return X, y, predictors

# =======================
# MODEL DEFINITIONS
# =======================

# 1. ANN
def create_ann_model_simple(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model

# Wrapper for Stacking
class ANN_Regressor_Wrapper(BaseEstimator, RegressorMixin):
    _estimator_type = "regressor"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    def __init__(self, epochs=100, batch_size=32):
        self.epochs = epochs

        self.batch_size = batch_size
        self.model = None
        self.input_dim = None

    def fit(self, X, y):
        self.input_dim = X.shape[1]
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(100, activation='relu', input_shape=(self.input_dim,)),
            tf.keras.layers.Dense(80, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict(self, X):
        return self.model.predict(X).flatten()

# 2. Random Forest
def create_rf_model():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=11,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=123,
        n_jobs=-1
    )

# 3. XGBoost
def create_xgb_model():
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=123,
        n_jobs=-1
    )

# 4. Meta-Model (Stacking)
def create_meta_model():
    ann = ANN_Regressor_Wrapper(epochs=100, batch_size=32)
    rf = create_rf_model()
    xgb = create_xgb_model()
    
    estimators = [('ANN', ann), ('RF', rf), ('XGB', xgb)]
    
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV()
    )
    return stacking_model

# =======================
# TRAINING & SAVING
# =======================
class UnifiedModelManager:
    """
    Manages all models including the new Physics-Informed ML (PIML) model.

    Model registry:
      - ANN   : Standard dense neural network
      - RF    : Random Forest
      - XGB   : XGBoost
      - Meta  : Stacking ensemble (ANN + RF + XGB → Ridge)
      - PIML  : Physics-Informed ML with electrochemistry constraints  ← NEW
    """

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.piml_scaler = None       # separate scaler for PIML (more features)
        self.piml_y_scaler = None     # target scaler for PIML
        self.piml_trainer = None      # PIMLTrainer instance
        self.is_fitted = False
        self.X_test = None
        self.y_test = None
        self.piml_data = None         # full data dict for PIML evaluation

    # ----------------------------------------------------------
    # TRAIN ALL BASELINE MODELS  (unchanged logic)
    # ----------------------------------------------------------
    def train_all(self):
        print("\nLoading and splitting data...")
        df = load_data()
        X, y, _ = preprocess_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        self.X_test = X_test
        self.y_test = y_test
        
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

        # 1. Train ANN
        print("\n--- Training ANN ---")
        ann = create_ann_model_simple(X_train_scaled.shape[1])
        ann.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)
        ann.save(os.path.join(MODEL_DIR, "ann_model.h5"))
        self.models["ANN"] = ann
        self._evaluate_baseline(ann, X_test_scaled, y_test, "ANN")

        # 2. Train RF
        print("\n--- Training Random Forest ---")
        rf = create_rf_model()
        rf.fit(X_train_scaled, y_train)
        joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))
        self.models["RF"] = rf
        self._evaluate_baseline(rf, X_test_scaled, y_test, "RF")

        # 3. Train XGB
        print("\n--- Training XGBoost ---")
        xgb = create_xgb_model()
        xgb.fit(X_train_scaled, y_train)
        joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))
        self.models["XGB"] = xgb
        self._evaluate_baseline(xgb, X_test_scaled, y_test, "XGB")

        # 4. Train Meta-Model
        print("\n--- Training Meta-Model (Stacking) ---")
        meta = create_meta_model()
        meta.fit(X_train_scaled, y_train)
        try:
            joblib.dump(meta, os.path.join(MODEL_DIR, "meta_model.pkl"))
        except Exception as e:
            print(f"Warning: Could not save Meta-model: {e}")
        self.models["Meta"] = meta
        self._evaluate_baseline(meta, X_test_scaled, y_test, "Meta")
        
        self.is_fitted = True
        print("\nAll baseline models trained and saved successfully!")

    # ----------------------------------------------------------
    # TRAIN PIML MODEL  ← NEW
    # ----------------------------------------------------------
    def train_piml(self, use_experimental: bool = True, epochs: int = 300,
                   lambda_faraday: float = 0.10, lambda_redox: float = 0.50,
                   lambda_butler_volmer: float = 0.05, lambda_nernst: float = 0.05,
                   lambda_charge_conservation: float = 0.05,
                   lambda_randles_sevcik: float = 0.05,
                   lambda_thermodynamic: float = 0.02):
        """
        Train the Physics-Informed ML model with 9 embedded
        electrochemical constraints (Faraday's law, capacitance,
        redox limits, smoothness, Butler-Volmer kinetics, Nernst
        equation, charge conservation, Randles-Sevcik, and
        thermodynamic consistency).
        """
        print("\n" + "=" * 55)
        print("  PHYSICS-INFORMED ML (PIML) MODEL TRAINING")
        print("=" * 55)

        # --- Load data ---
        if use_experimental:
            try:
                df = load_experimental_data()
                print("Using experimental dataset.")
            except FileNotFoundError:
                print("Experimental data not found. Generating synthetic dataset...")
                df = generate_synthetic_dataset(
                    n_potential_points=200, noise_std=0.03,
                    save_path=os.path.join("Dataset", "synthetic_cv_dataset.csv"),
                )
        else:
            csv_path = os.path.join("Dataset", "synthetic_cv_dataset.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                df = generate_synthetic_dataset(
                    n_potential_points=200, noise_std=0.03, save_path=csv_path,
                )
            print("Using synthetic dataset.")

        # --- Select features ---
        available = set(df.columns)
        feature_cols = [c for c in PIML_FEATURE_COLS if c in available]
        target_col = "Current_Density" if "Current_Density" in available else "Current"
        print(f"Features ({len(feature_cols)}): {feature_cols}")
        print(f"Target: {target_col}")

        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)

        # --- Three-way split ---
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)
        print(f"Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

        # --- Scale ---
        self.piml_scaler = StandardScaler()
        X_train_s = self.piml_scaler.fit_transform(X_train).astype(np.float32)
        X_val_s   = self.piml_scaler.transform(X_val).astype(np.float32)
        X_test_s  = self.piml_scaler.transform(X_test).astype(np.float32)
        joblib.dump(self.piml_scaler, os.path.join(MODEL_DIR, "piml_scaler.pkl"))

        # --- Scale target ---
        self.piml_y_scaler = StandardScaler()
        y_train_s = self.piml_y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
        y_val_s   = self.piml_y_scaler.transform(y_val.reshape(-1, 1)).flatten().astype(np.float32)
        y_test_s  = self.piml_y_scaler.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)
        joblib.dump(self.piml_y_scaler, os.path.join(MODEL_DIR, "piml_y_scaler.pkl"))

        # --- Build model ---
        model = build_piml_ann(
            input_dim=X_train_s.shape[1],
            hidden_units=[256, 128, 64, 32],
            dropout_rate=0.10,
            mc_dropout=True,
        )
        model.summary()

        physics_loss = PhysicsLoss(
            lambda_faraday=lambda_faraday,
            lambda_capacitance=0.10,
            lambda_redox=lambda_redox,
            lambda_smooth=0.05,
            lambda_butler_volmer=lambda_butler_volmer,
            lambda_nernst=lambda_nernst,
            lambda_charge_conservation=lambda_charge_conservation,
            lambda_randles_sevcik=lambda_randles_sevcik,
            lambda_thermodynamic=lambda_thermodynamic,
            y_mean=float(self.piml_y_scaler.mean_[0]),
            y_scale=float(self.piml_y_scaler.scale_[0]),
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.piml_trainer = PIMLTrainer(model, physics_loss, optimizer, feature_cols)

        # --- Train ---
        history = self.piml_trainer.fit(
            X_train_s, X_train, y_train_s,
            X_val_s, X_val, y_val_s,
            epochs=epochs, batch_size=512, patience=30, verbose=1,
        )

        # Save
        model.save(os.path.join(MODEL_DIR, "piml_model.keras"))
        hist_path = os.path.join(MODEL_DIR, "piml_history.json")
        with open(hist_path, "w") as f:
            json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)

        self.models["PIML"] = model
        self.piml_data = {
            "X_train_scaled": X_train_s, "X_train_raw": X_train,
            "X_val_scaled": X_val_s,     "X_val_raw": X_val,
            "X_test_scaled": X_test_s,   "X_test_raw": X_test,
            "y_train": y_train_s, "y_val": y_val_s, "y_test": y_test_s,
            "y_train_raw": y_train, "y_val_raw": y_val, "y_test_raw": y_test,
            "scaler": self.piml_scaler, "y_scaler": self.piml_y_scaler,
            "feature_cols": feature_cols, "df": df,
        }

        # Quick evaluation
        print("\n===== PIML TEST EVALUATION =====")
        y_mean, y_std = self.piml_trainer.predict(X_test_s, n_mc_samples=50)
        # Inverse-transform to original scale
        y_mean_orig = self.piml_y_scaler.inverse_transform(y_mean.reshape(-1, 1)).flatten()
        y_std_orig = y_std * self.piml_y_scaler.scale_[0] if y_std is not None else None
        metrics = compute_metrics(y_test, y_mean_orig)
        violations = compute_constraint_violations(y_mean_orig, X_test)
        print(f"  R²   = {metrics['R2']:.5f}")
        print(f"  RMSE = {metrics['RMSE']:.5f}")
        print(f"  MAE  = {metrics['MAE']:.5f}")
        for k, v in violations.items():
            print(f"  {k}: {v}%")

        self.is_fitted = True
        print("\nPIML model trained and saved!")

    # ----------------------------------------------------------
    # FULL PIML EVALUATION  ← NEW
    # ----------------------------------------------------------
    def evaluate_piml_full(self):
        """Run the complete evaluation suite on the PIML model."""
        if self.piml_trainer is None or self.piml_data is None:
            print("PIML model not trained/loaded. Train or load first.")
            return

        cfg = {
            "plot_dir": PLOT_DIR, "model_dir": MODEL_DIR,
            "mc_samples": 50, "mc_dropout": True, "max_current": 50.0,
        }
        evaluate_piml(self.piml_trainer, self.piml_data, cfg)

    # ----------------------------------------------------------
    # BASELINE EVALUATE
    # ----------------------------------------------------------
    def _evaluate_baseline(self, model, X_test, y_test, name):
        preds = model.predict(X_test)
        if hasattr(preds, 'shape') and len(preds.shape) > 1:
            preds = preds.flatten()
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"[{name}] Test MSE: {mse:.4f}, R2: {r2:.4f}")
        plt.figure(figsize=(5, 5))
        plt.scatter(y_test, preds, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.title(f"{name} Actual vs Predicted")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{name}_results.png"))
        plt.close()

    # ----------------------------------------------------------
    # LOAD MODELS
    # ----------------------------------------------------------
    def load_models(self):
        print("\nLoading models from disk...")
        try:
            scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            if os.path.exists(os.path.join(MODEL_DIR, "ann_model.h5")):
                self.models["ANN"] = tf.keras.models.load_model(
                    os.path.join(MODEL_DIR, "ann_model.h5"))

            if os.path.exists(os.path.join(MODEL_DIR, "rf_model.pkl")):
                self.models["RF"] = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))

            if os.path.exists(os.path.join(MODEL_DIR, "xgb_model.pkl")):
                self.models["XGB"] = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))

            if os.path.exists(os.path.join(MODEL_DIR, "meta_model.pkl")):
                self.models["Meta"] = joblib.load(os.path.join(MODEL_DIR, "meta_model.pkl"))

            # Load PIML model
            piml_path = os.path.join(MODEL_DIR, "piml_model.keras")
            piml_scaler_path = os.path.join(MODEL_DIR, "piml_scaler.pkl")
            piml_y_scaler_path = os.path.join(MODEL_DIR, "piml_y_scaler.pkl")
            if os.path.exists(piml_path):
                from PIML_model import MCDropout
                piml_model = tf.keras.models.load_model(
                    piml_path, custom_objects={"MCDropout": MCDropout})
                self.models["PIML"] = piml_model
                if os.path.exists(piml_scaler_path):
                    self.piml_scaler = joblib.load(piml_scaler_path)
                if os.path.exists(piml_y_scaler_path):
                    self.piml_y_scaler = joblib.load(piml_y_scaler_path)
                # Reconstruct trainer for prediction
                y_m = float(self.piml_y_scaler.mean_[0]) if self.piml_y_scaler else 0.0
                y_s = float(self.piml_y_scaler.scale_[0]) if self.piml_y_scaler else 1.0
                physics_loss = PhysicsLoss(y_mean=y_m, y_scale=y_s)
                self.piml_trainer = PIMLTrainer(piml_model, physics_loss)

            if not self.models:
                print("No models found. Please train models first.")
                return False

            print(f"Loaded: {list(self.models.keys())}")
            self.is_fitted = True
            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    # ----------------------------------------------------------
    # PREDICT
    # ----------------------------------------------------------
    def predict_single(self, model_name, inputs, n_mc_samples=0):
        """
        Predict current density for a single input.
        For PIML model, optionally returns uncertainty via MC Dropout.
        """
        if not self.is_fitted:
            print("Models not loaded/trained.")
            return None

        if model_name not in self.models:
            print(f"Model {model_name} not available. Available: {list(self.models.keys())}")
            return None

        if model_name == "PIML":
            if self.piml_scaler is None:
                print("PIML scaler not loaded.")
                return None
            inputs_scaled = self.piml_scaler.transform(inputs).astype(np.float32)
            if self.piml_trainer and n_mc_samples > 0:
                y_mean, y_std = self.piml_trainer.predict(inputs_scaled, n_mc_samples)
                # Inverse-transform to original scale
                if self.piml_y_scaler is not None:
                    y_mean = self.piml_y_scaler.inverse_transform(y_mean.reshape(-1, 1)).flatten()
                    y_std = y_std * self.piml_y_scaler.scale_[0] if y_std is not None else None
                val = y_mean[0]
                std_val = y_std[0] if y_std is not None else 0.0
                return val, std_val
            pred = self.models["PIML"].predict(inputs_scaled, verbose=0).flatten()
            if self.piml_y_scaler is not None:
                pred = self.piml_y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
            return pred[0]
        else:
            inputs_scaled = self.scaler.transform(inputs)
            model = self.models[model_name]
            pred = model.predict(inputs_scaled)
            if isinstance(pred, np.ndarray):
                return pred.flatten()[0]
            return pred


# =======================
# MENU INTERFACE
# =======================

def get_prediction_input(model_name="ANN"):
    """
    Gather prediction input from user.
    For PIML model, also collects Temperature and Electrode Area.
    """
    print("\n--- Enter Prediction Parameters ---")
    try:
        potential = float(input("Enter Potential (V): "))
        scan_rate = float(input("Enter Scan Rate (mV/s): "))

        print("\nSelect Material Composition:")
        print("1. BFO Pure")
        print("2. BFZO (Zn doped)")
        print("3. BFCO (Co doped)")
        print("4. Manual Entry")
        mat_choice = input("Select (1-4): ")

        zn = 0; co = 0; conc = 0.0
        if mat_choice == '1':
            pass
        elif mat_choice == '2':
            zn = 1
            conc = float(input("Enter Zn Doping Level (e.g. 0.10): "))
        elif mat_choice == '3':
            co = 1
            conc = float(input("Enter Co Doping Level: "))
        elif mat_choice == '4':
            zn = int(input("ZN (1/0): "))
            co = int(input("CO (1/0): "))
            conc = float(input("Concentration: "))

        oxid = int(input("Oxidation State (1=Oxid, 0=Red): "))

        if model_name == "PIML":
            temperature = float(input("Temperature (K) [default 298]: ") or "298")
            electrode_area = float(input("Electrode Area (cm²) [default 0.07]: ") or "0.07")
            # PIML features: Potential, OXIDATION, Zn_Co_Conc, Scan_Rate, ZN, CO, Temperature, Electrode_Area
            features = np.array([[potential, oxid, conc, scan_rate, zn, co,
                                  temperature, electrode_area]], dtype=np.float32)
        else:
            # Legacy features: Potential, OXIDATION, Zn/Co_Conc, SCAN_RATE, ZN, CO
            features = np.array([[potential, oxid, conc, scan_rate, zn, co]])

        return features

    except ValueError:
        print("Invalid input.")
        return None


def main():
    manager = UnifiedModelManager()

    while True:
        print("\n" + "=" * 55)
        print("  UNIFIED MODEL INTERFACE  –  Physics-Informed ML")
        print("  9 Electrochemical Constraints | Hybrid AI + Physics")
        print("=" * 55)
        print("1. Train Baseline Models (ANN, RF, XGB, Meta)")
        print("2. Train PIML Model (Physics-Informed, 9 Constraints)")
        print("3. Load Existing Models")
        print("4. Predict Outcome")
        print("5. Full PIML Evaluation & Visualisation")
        print("6. Find Optimal Zn/Co Composition (PIML)")
        print("7. Extrapolation Analysis (PIML)")
        print("8. Compare All Models (PIML vs Baselines)")
        print("9. Butler-Volmer & Physics Validation")
        print("10. Generate Physics Report")
        print("11. Exit")

        choice = input("\nSelect Option (1-11): ").strip()

        if choice == '1':
            manager.train_all()

        elif choice == '2':
            src = input("Data source? (1) Experimental  (2) Synthetic [default: 1]: ").strip()
            use_exp = src != '2'
            ep = input("Training epochs [default 300]: ").strip()
            epochs = int(ep) if ep else 300
            manager.train_piml(use_experimental=use_exp, epochs=epochs)

        elif choice == '3':
            manager.load_models()

        elif choice == '4':
            if not manager.is_fitted:
                print("!! You must Load or Train models first !!")
                continue

            print("\nAvailable Models:", list(manager.models.keys()))
            m_choice = input("Choose Model (ANN/RF/XGB/Meta/PIML) [Default: PIML]: ").strip()
            if not m_choice:
                m_choice = "PIML" if "PIML" in manager.models else "Meta"

            input_data = get_prediction_input(model_name=m_choice)
            if input_data is not None:
                if m_choice == "PIML":
                    result = manager.predict_single(m_choice, input_data, n_mc_samples=50)
                    if isinstance(result, tuple):
                        mean_val, std_val = result
                        print(f"\n>> PIML Predicted Current Density: {mean_val:.6f} mA/cm²")
                        print(f"   Uncertainty (±1.96σ):           ±{1.96 * std_val:.6f} mA/cm²")
                    elif result is not None:
                        print(f"\n>> PIML Predicted Current Density: {result:.6f} mA/cm²")
                else:
                    prediction = manager.predict_single(m_choice, input_data)
                    if prediction is not None:
                        print(f"\n>> {m_choice} Predicted Current: {prediction:.6f}")

        elif choice == '5':
            manager.evaluate_piml_full()

        elif choice == '6':
            if "PIML" not in manager.models or manager.piml_scaler is None:
                print("PIML model not available. Train or load first.")
                continue
            feature_cols = PIML_FEATURE_COLS
            analyse_optimal_composition(
                manager.models["PIML"],
                manager.piml_scaler,
                feature_cols[:manager.models["PIML"].input_shape[-1]],
                save_path=os.path.join(PLOT_DIR, "PIML_optimal_composition.png"),
            )

        elif choice == '7':
            if "PIML" not in manager.models or manager.piml_scaler is None:
                print("PIML model not available. Train or load first.")
                continue
            feature_cols = PIML_FEATURE_COLS[:manager.models["PIML"].input_shape[-1]]
            extrap_results = extrapolation_analysis(
                manager.models["PIML"],
                manager.piml_scaler,
                feature_cols,
                save_path=os.path.join(PLOT_DIR, "PIML_extrapolation.png"),
            )
            print(f"\nRandles-Sevcik R²: {extrap_results.get('randles_sevcik_r2', 0):.4f}")

        elif choice == '8':
            if not manager.models:
                print("No models available. Train or load first.")
                continue
            if manager.piml_data is not None:
                compare_all_models(
                    models_dict=manager.models,
                    X_test=manager.piml_data["X_test_raw"],
                    y_test=manager.piml_data["y_test_raw"],
                    X_test_scaled_piml=manager.piml_data["X_test_scaled"],
                    X_test_raw=manager.piml_data["X_test_raw"],
                    piml_trainer=manager.piml_trainer,
                    piml_y_scaler=manager.piml_y_scaler,
                    save_dir=PLOT_DIR,
                )
            else:
                print("Train PIML model first to enable full comparison.")

        elif choice == '9':
            if "PIML" not in manager.models or manager.piml_scaler is None:
                print("PIML model not available. Train or load first.")
                continue
            feature_cols = PIML_FEATURE_COLS[:manager.models["PIML"].input_shape[-1]]
            plot_butler_volmer_adherence(
                manager.models["PIML"],
                manager.piml_scaler,
                feature_cols,
                save_path=os.path.join(PLOT_DIR, "PIML_butler_volmer_analysis.png"),
                y_scaler=manager.piml_y_scaler,
            )
            if manager.piml_data is not None:
                y_mean, y_std = manager.piml_trainer.predict(
                    manager.piml_data["X_test_scaled"].astype(np.float32), n_mc_samples=50)
                y_mean_orig = manager.piml_y_scaler.inverse_transform(
                    y_mean.reshape(-1, 1)).flatten()
                y_test = manager.piml_data["y_test_raw"]
                metrics = compute_metrics(y_test, y_mean_orig)
                violations = compute_constraint_violations(
                    y_mean_orig, manager.piml_data["X_test_raw"])
                plot_physics_dashboard(
                    manager.models["PIML"],
                    manager.piml_scaler,
                    feature_cols,
                    y_pred=y_mean_orig,
                    y_true=y_test,
                    X_raw=manager.piml_data["X_test_raw"],
                    violations=violations,
                    metrics=metrics,
                    save_path=os.path.join(PLOT_DIR, "PIML_physics_dashboard.png"),
                    y_scaler=manager.piml_y_scaler,
                )

        elif choice == '10':
            if manager.piml_data is not None and manager.piml_trainer is not None:
                feature_cols = PIML_FEATURE_COLS[:manager.models["PIML"].input_shape[-1]]
                y_mean, _ = manager.piml_trainer.predict(
                    manager.piml_data["X_test_scaled"].astype(np.float32), n_mc_samples=50)
                y_mean_orig = manager.piml_y_scaler.inverse_transform(
                    y_mean.reshape(-1, 1)).flatten()
                y_test = manager.piml_data["y_test_raw"]
                metrics = compute_metrics(y_test, y_mean_orig)
                violations = compute_constraint_violations(
                    y_mean_orig, manager.piml_data["X_test_raw"])
                generate_physics_report(
                    metrics, violations,
                    save_path=os.path.join(PLOT_DIR, "PIML_physics_report.txt"),
                )
            else:
                print("Train PIML model first.")

        elif choice == '11':
            print("Exiting...")
            break
        else:
            print("Invalid Option.")


if __name__ == "__main__":
    main()
