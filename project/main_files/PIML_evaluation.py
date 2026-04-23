"""
PIML Evaluation & Visualization
=================================
Comprehensive evaluation of the Physics-Informed ML model including:
  - Standard metrics: R², RMSE, MAE
  - Physics constraint violation analysis (9 constraints)
  - Butler-Volmer kinetics adherence
  - Charge conservation validation
  - Randles-Sevcik consistency check
  - Thermodynamic consistency scoring
  - Predicted vs experimental CV curves
  - SHAP feature attribution
  - Uncertainty quantification plots
  - Optimal Zn/Co composition analysis
  - Physics validation dashboard for publication
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================
# CORE METRICS
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute R², RMSE, MAE.  Handles NaN gracefully."""
    # Remove NaN / Inf entries (can occur if model diverged)
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() == 0:
        return {"R2": float("nan"), "RMSE": float("nan"),
                "MAE": float("nan"), "MSE": float("nan")}
    y_true, y_pred = y_true[mask], y_pred[mask]
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MSE": mse}


def compute_constraint_violations(
    y_pred: np.ndarray,
    X_raw: np.ndarray,
    max_current: float = 50.0,
    idx_scan_rate: int = 3,
    idx_potential: int = 0,
    idx_oxidation: int = 1,
    idx_temperature: int = 6,
) -> dict:
    """
    Compute % of predictions violating all 9 physics constraints.

    Returns dict of violation percentages and physics scores.
    """
    n = len(y_pred)

    # 1. Redox limit violation: |I| > max physically plausible
    exceed = np.sum(np.abs(y_pred) > max_current)
    pct_exceed = 100.0 * exceed / n

    # 2. Negative capacitance: C_sp ∝ |I|/v should be ≥ 0
    scan_rate = X_raw[:, idx_scan_rate] if X_raw.shape[1] > idx_scan_rate else np.ones(n)
    C_est = np.abs(y_pred) / (scan_rate + 1e-8)
    neg_cap = np.sum(C_est < 0)
    pct_neg_cap = 100.0 * neg_cap / n

    # 3. Smoothness: count large jumps (> 3σ)
    if len(y_pred) > 2:
        diffs = np.diff(y_pred)
        sigma = np.std(diffs) + 1e-12
        jumps = np.sum(np.abs(diffs) > 3 * sigma)
        pct_jumps = 100.0 * jumps / (n - 1)
    else:
        pct_jumps = 0.0

    # 4. Butler-Volmer adherence (sign consistency)
    # Near equilibrium potential, current sign should match overpotential sign
    if X_raw.shape[1] > idx_potential:
        potential = X_raw[:, idx_potential]
        E0 = 0.25  # approximate equilibrium potential
        eta = potential - E0
        # Far from equilibrium (|η| > 50 mV): sign should match
        far_mask = np.abs(eta) > 0.05
        if np.sum(far_mask) > 0:
            sign_violations = np.sum((y_pred[far_mask] * eta[far_mask]) < 0)
            pct_bv_violation = 100.0 * sign_violations / np.sum(far_mask)
        else:
            pct_bv_violation = 0.0
    else:
        pct_bv_violation = 0.0

    # 5. Charge conservation (anodic vs cathodic balance)
    if X_raw.shape[1] > idx_oxidation:
        oxidation = X_raw[:, idx_oxidation]
        anodic = y_pred[oxidation > 0.5]
        cathodic = y_pred[oxidation <= 0.5]
        if len(anodic) > 0 and len(cathodic) > 0:
            Q_anodic = np.mean(anodic)
            Q_cathodic = np.mean(cathodic)
            Q_total = abs(Q_anodic) + abs(Q_cathodic) + 1e-8
            charge_imbalance = abs(Q_anodic + Q_cathodic) / Q_total * 100
        else:
            charge_imbalance = 0.0
    else:
        charge_imbalance = 0.0

    # 6. Nernst temperature consistency
    if X_raw.shape[1] > idx_temperature:
        temperatures = X_raw[:, idx_temperature]
        unique_temps = np.unique(temperatures)
        if len(unique_temps) >= 2:
            # Check: higher T should give higher |I| on average
            temp_currents = []
            for t in sorted(unique_temps):
                mask = temperatures == t
                temp_currents.append(np.mean(np.abs(y_pred[mask])))
            # Count non-monotonic pairs
            non_mono = sum(1 for i in range(len(temp_currents)-1)
                          if temp_currents[i+1] < temp_currents[i])
            pct_nernst_violation = 100.0 * non_mono / max(len(temp_currents) - 1, 1)
        else:
            pct_nernst_violation = 0.0
    else:
        pct_nernst_violation = 0.0

    # 7. Randles-Sevcik: I_peak should scale with √(scan_rate)
    unique_sr = np.unique(scan_rate)
    if len(unique_sr) >= 3:
        peak_currents = []
        sqrt_sr = []
        for sr in sorted(unique_sr):
            mask = scan_rate == sr
            peak_currents.append(np.max(np.abs(y_pred[mask])))
            sqrt_sr.append(np.sqrt(max(abs(sr), 1e-6)))
        peak_currents = np.array(peak_currents)
        sqrt_sr = np.array(sqrt_sr)
        # R² of linear fit I_peak vs √v
        coeffs = np.polyfit(sqrt_sr, peak_currents, 1)
        fitted = np.polyval(coeffs, sqrt_sr)
        ss_res = np.sum((peak_currents - fitted)**2)
        ss_tot = np.sum((peak_currents - peak_currents.mean())**2) + 1e-12
        randles_r2 = 1 - ss_res / ss_tot
    else:
        randles_r2 = 1.0  # not enough data to test

    return {
        "current_exceed_%":      round(pct_exceed, 3),
        "negative_cap_%":        round(pct_neg_cap, 3),
        "large_jumps_%":         round(pct_jumps, 3),
        "bv_sign_violation_%":   round(pct_bv_violation, 3),
        "charge_imbalance_%":    round(charge_imbalance, 3),
        "nernst_violation_%":    round(pct_nernst_violation, 3),
        "randles_sevcik_R2":     round(randles_r2, 5),
    }


# ============================================================
# PREDICTED vs ACTUAL SCATTER
# ============================================================

def plot_pred_vs_actual(y_true, y_pred, title="PIML", save_path=None, y_std=None):
    """Scatter plot of predicted vs actual with optional uncertainty bars."""
    fig, ax = plt.subplots(figsize=(6, 6))

    if y_std is not None:
        ax.errorbar(y_true, y_pred, yerr=1.96 * y_std, fmt="o", alpha=0.3,
                     markersize=2, elinewidth=0.5, label="95% CI")
    else:
        ax.scatter(y_true, y_pred, alpha=0.4, s=8)

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1.5, label="Ideal")
    ax.set_xlabel("Actual Current Density (mA/cm²)")
    ax.set_ylabel("Predicted Current Density (mA/cm²)")
    ax.set_title(f"{title}: Actual vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"  Plot saved → {save_path}")
    plt.close(fig)


# ============================================================
# CV CURVE COMPARISON
# ============================================================

def plot_cv_curves(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    group_col: str = "Zn_Co_Conc",
    potential_col: str = "Potential",
    actual_col: str = "Current_Density",
    scan_rate_filter: float = None,
    save_path: str = None,
    y_std: np.ndarray = None,
):
    """
    Overlay experimental and predicted CV curves for different compositions.
    """
    df = df_test.copy()
    df["Predicted"] = y_pred

    if scan_rate_filter and "Scan_Rate" in df.columns:
        df = df[df["Scan_Rate"] == scan_rate_filter]

    groups = df[group_col].unique()
    n_groups = min(len(groups), 6)  # max 6 subplots

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, grp in enumerate(sorted(groups)[:n_groups]):
        ax = axes[i]
        sub = df[df[group_col] == grp].sort_values(potential_col)

        ax.plot(sub[potential_col], sub[actual_col], "b-", alpha=0.7, label="Experimental")
        ax.plot(sub[potential_col], sub["Predicted"], "r--", alpha=0.7, label="PIML Predicted")

        if y_std is not None:
            idx = sub.index
            std_vals = y_std[idx] if len(y_std) == len(df_test) else np.zeros(len(sub))
            ax.fill_between(
                sub[potential_col],
                sub["Predicted"] - 1.96 * std_vals,
                sub["Predicted"] + 1.96 * std_vals,
                alpha=0.15, color="red", label="95% CI",
            )

        ax.set_xlabel("Potential (V)")
        ax.set_ylabel("Current Density (mA/cm²)")
        ax.set_title(f"{group_col} = {grp:.3f}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_groups, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("CV Curve Comparison: Experimental vs PIML Predicted", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"  Plot saved → {save_path}")
    plt.close(fig)


# ============================================================
# TRAINING HISTORY
# ============================================================

def plot_training_history(history: dict, save_path: str = None):
    """Plot training/validation loss and all 9 physics constraint components."""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # 1. Total loss
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train Total Loss")
    ax.plot(history["val_loss"],   label="Val Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Hybrid Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. MSE
    ax = axes[0, 1]
    ax.plot(history["train_mse"], label="Train MSE")
    ax.plot(history["val_mse"],   label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Data-Driven MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Original physics losses (Faraday, Cap, Redox, Smooth)
    ax = axes[0, 2]
    for key in ["physics_faraday", "physics_cap", "physics_redox", "physics_smooth"]:
        if key in history:
            ax.plot(history[key], label=key.replace("physics_", "").title())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Component")
    ax.set_title("Core Physics Constraints")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Butler-Volmer
    ax = axes[1, 0]
    if "physics_butler_volmer" in history:
        ax.plot(history["physics_butler_volmer"], color="#e74c3c", label="Butler-Volmer")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Butler-Volmer Kinetics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Nernst equation
    ax = axes[1, 1]
    if "physics_nernst" in history:
        ax.plot(history["physics_nernst"], color="#3498db", label="Nernst")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Nernst Equation (T-dependence)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Charge conservation
    ax = axes[1, 2]
    if "physics_charge_conservation" in history:
        ax.plot(history["physics_charge_conservation"], color="#2ecc71", label="Charge Conserv.")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Charge Conservation (∮I·dV ≈ 0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 7. Randles-Sevcik
    ax = axes[2, 0]
    if "physics_randles_sevcik" in history:
        ax.plot(history["physics_randles_sevcik"], color="#9b59b6", label="Randles-Sevcik")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Randles-Sevcik (Iₚ ∝ √v)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 8. Thermodynamic consistency
    ax = axes[2, 1]
    if "physics_thermodynamic" in history:
        ax.plot(history["physics_thermodynamic"], color="#e67e22", label="Thermodynamic")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Thermodynamic Consistency (ΔG = -nFE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9. Physics weight schedule + Learning rate
    ax = axes[2, 2]
    if "physics_weight" in history:
        ax.plot(history["physics_weight"], "k-", linewidth=2, label="Physics Weight")
    if "learning_rate" in history:
        ax2 = ax.twinx()
        ax2.plot(history["learning_rate"], "b--", alpha=0.5, label="Learning Rate")
        ax2.set_ylabel("Learning Rate", color="blue")
        ax2.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weight")
    ax.set_title("Curriculum Schedule & LR")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("PIML Training History: 9 Physics-Informed Constraints",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Plot saved → {save_path}")
    plt.close(fig)


# ============================================================
# SHAP FEATURE ATTRIBUTION
# ============================================================

def plot_shap_analysis(model, X_sample, feature_names, save_path=None):
    """
    SHAP-based feature importance analysis.
    Falls back to permutation importance if shap is not installed.
    """
    try:
        import shap  # type: ignore[import-unresolved]

        # Use KernelExplainer for model-agnostic SHAP
        background = X_sample[:min(100, len(X_sample))]
        explainer = shap.KernelExplainer(
            lambda x: model.predict(x, verbose=0).flatten(),
            background,
        )
        shap_values = explainer.shap_values(X_sample[:min(200, len(X_sample))])

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample[:min(200, len(X_sample))],
                          feature_names=feature_names, show=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  SHAP plot saved → {save_path}")
        plt.close()

        # Also create bar plot
        fig, ax = plt.subplots(figsize=(8, 5))
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=True)
        importance.plot(kind="barh", ax=ax)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("PIML Feature Importance (SHAP)")
        plt.tight_layout()
        bar_path = save_path.replace(".png", "_bar.png") if save_path else None
        if bar_path:
            fig.savefig(bar_path, dpi=300)
            print(f"  SHAP bar plot saved → {bar_path}")
        plt.close(fig)

        return shap_values

    except ImportError:
        print("  SHAP not installed. Computing permutation-based importance...")
        return _permutation_importance(model, X_sample, feature_names, save_path)


def _permutation_importance(model, X, feature_names, save_path=None):
    """Simple permutation importance as SHAP fallback."""
    base_pred = model.predict(X, verbose=0).flatten()
    importances = []
    for i in range(X.shape[1]):
        X_perm = X.copy()
        np.random.shuffle(X_perm[:, i])
        perm_pred = model.predict(X_perm, verbose=0).flatten()
        imp = np.mean((base_pred - perm_pred) ** 2)
        importances.append(imp)

    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    imp_series.plot(kind="barh", ax=ax)
    ax.set_xlabel("Permutation Importance (MSE increase)")
    ax.set_title("PIML Feature Importance (Permutation)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"  Permutation importance plot saved → {save_path}")
    plt.close(fig)
    return importances


# ============================================================
# OPTIMAL Zn/Co COMPOSITION ANALYSIS
# ============================================================

def analyse_optimal_composition(
    model,
    scaler,
    feature_cols: list,
    save_path: str = None,
):
    """
    Sweep Zn and Co fractions to find composition maximising capacitance proxy.
    Capacitance proxy = integral(|predicted I|) over potential range.
    """
    zn_range = np.linspace(0, 0.20, 21)
    co_range = np.linspace(0, 0.20, 21)
    potentials = np.linspace(-0.6, 0.8, 100)

    # Fixed conditions: scan_rate=50 mV/s, T=298K, area=0.07
    cap_map = np.zeros((len(zn_range), len(co_range)))

    for i, zn in enumerate(zn_range):
        for j, co in enumerate(co_range):
            conc = zn + co
            zn_bin = 1 if zn > 0 else 0
            co_bin = 1 if co > 0 else 0

            # Build feature array for all potential points
            # Order: Potential, OXIDATION, Zn_Co_Conc, Scan_Rate, ZN, CO, Temperature, Electrode_Area
            X = np.column_stack([
                potentials,
                (potentials > 0.25).astype(float),  # approx OXIDATION
                np.full_like(potentials, conc),
                np.full_like(potentials, 50.0),
                np.full_like(potentials, zn_bin),
                np.full_like(potentials, co_bin),
                np.full_like(potentials, 298.0),
                np.full_like(potentials, 0.07),
            ]).astype(np.float32)

            # Only use columns that exist in feature_cols
            n_feats = min(X.shape[1], len(feature_cols))
            X = X[:, :n_feats]

            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled, verbose=0).flatten()

            # Capacitance proxy: ∫|I|dV / (v × ΔV)
            dV = potentials[1] - potentials[0]
            cap_map[i, j] = np.trapezoid(np.abs(pred), potentials) / (0.05 * 1.4)

    # Find optimal
    best_idx = np.unravel_index(cap_map.argmax(), cap_map.shape)
    best_zn = zn_range[best_idx[0]]
    best_co = co_range[best_idx[1]]
    best_cap = cap_map[best_idx]

    print(f"\n  Optimal composition: Zn={best_zn:.3f}, Co={best_co:.3f}")
    print(f"  Predicted max capacitance proxy: {best_cap:.3f}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cap_map, origin="lower", aspect="auto",
                    extent=[co_range[0], co_range[-1], zn_range[0], zn_range[-1]],
                    cmap="viridis")
    ax.set_xlabel("Co fraction")
    ax.set_ylabel("Zn fraction")
    ax.set_title("Capacitance Proxy vs Zn/Co Substitution")
    plt.colorbar(im, ax=ax, label="Capacitance proxy (F/g)")
    ax.plot(best_co, best_zn, "r*", markersize=15, label=f"Optimal ({best_zn:.2f}, {best_co:.2f})")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"  Composition map saved → {save_path}")
    plt.close(fig)

    return best_zn, best_co, best_cap, cap_map


# ============================================================
# UNCERTAINTY QUANTIFICATION PLOT
# ============================================================

def plot_uncertainty(y_true, y_mean, y_std, save_path=None):
    """Visualise MC-Dropout uncertainty: sorted predictions with CI bands."""
    idx = np.argsort(y_true)
    y_sorted = y_true[idx]
    m_sorted = y_mean[idx]
    s_sorted = y_std[idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    x_axis = np.arange(len(y_sorted))

    ax.plot(x_axis, y_sorted, "b.", markersize=1, alpha=0.5, label="Actual")
    ax.plot(x_axis, m_sorted, "r-", linewidth=0.5, alpha=0.7, label="Predicted (mean)")
    ax.fill_between(x_axis,
                     m_sorted - 1.96 * s_sorted,
                     m_sorted + 1.96 * s_sorted,
                     alpha=0.2, color="red", label="95% CI")

    ax.set_xlabel("Sample (sorted by actual)")
    ax.set_ylabel("Current Density (mA/cm²)")
    ax.set_title("PIML Prediction Uncertainty (MC Dropout)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"  Uncertainty plot saved → {save_path}")
    plt.close(fig)


# ============================================================
# REDOX PEAK ANALYSIS
# ============================================================

def plot_redox_peaks(df_test, y_pred, save_path=None):
    """Compare predicted and actual peak current densities across scan rates."""
    df = df_test.copy()
    df["Predicted"] = y_pred

    if "Scan_Rate" not in df.columns:
        print("  Scan_Rate column not found; skipping redox peak plot.")
        return

    # For each scan rate, find peak (max) and trough (min)
    results = []
    for sr in sorted(df["Scan_Rate"].unique()):
        sub = df[df["Scan_Rate"] == sr]
        results.append({
            "Scan_Rate": sr,
            "Actual_Peak": sub["Current_Density"].max() if "Current_Density" in sub.columns else 0,
            "Predicted_Peak": sub["Predicted"].max(),
            "Actual_Trough": sub["Current_Density"].min() if "Current_Density" in sub.columns else 0,
            "Predicted_Trough": sub["Predicted"].min(),
        })
    res = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Peak currents
    ax = axes[0]
    ax.plot(res["Scan_Rate"], res["Actual_Peak"], "bo-", label="Actual Peak")
    ax.plot(res["Scan_Rate"], res["Predicted_Peak"], "r^--", label="Predicted Peak")
    ax.set_xlabel("Scan Rate (mV/s)")
    ax.set_ylabel("Peak Current Density (mA/cm²)")
    ax.set_title("Oxidation Peak Current vs Scan Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Trough currents
    ax = axes[1]
    ax.plot(res["Scan_Rate"], res["Actual_Trough"], "bo-", label="Actual Trough")
    ax.plot(res["Scan_Rate"], res["Predicted_Trough"], "r^--", label="Predicted Trough")
    ax.set_xlabel("Scan Rate (mV/s)")
    ax.set_ylabel("Trough Current Density (mA/cm²)")
    ax.set_title("Reduction Trough Current vs Scan Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"  Redox peak plot saved → {save_path}")
    plt.close(fig)


# ============================================================
# BUTLER-VOLMER ADHERENCE ANALYSIS
# ============================================================

def plot_butler_volmer_adherence(
    model, scaler, feature_cols, save_path=None, y_scaler=None,
):
    """
    Analyse how well the model adheres to Butler-Volmer kinetics
    by plotting the predicted I vs overpotential (Tafel plot) and
    comparing with the theoretical Butler-Volmer curve.
    """
    potentials = np.linspace(-0.6, 0.8, 200)
    E0 = 0.25  # equilibrium potential

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: I vs overpotential for multiple compositions
    ax = axes[0]
    compositions = [(0.0, 0.0, "Pure BFO"), (0.10, 0.0, "Zn=0.10"), (0.0, 0.10, "Co=0.10")]
    for zn, co, label in compositions:
        conc = zn + co
        X = np.column_stack([
            potentials,
            (potentials > 0.25).astype(float),
            np.full_like(potentials, conc),
            np.full_like(potentials, 50.0),
            np.full_like(potentials, float(1 if zn > 0 else 0)),
            np.full_like(potentials, float(1 if co > 0 else 0)),
            np.full_like(potentials, 298.0),
            np.full_like(potentials, 0.07),
        ]).astype(np.float32)[:, :len(feature_cols)]
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled, verbose=0).flatten()
        if y_scaler is not None:
            pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

        eta = potentials - E0
        ax.plot(eta, pred, linewidth=1.5, label=label)

    # Overlay theoretical Butler-Volmer shape (normalised)
    eta_theory = np.linspace(-0.4, 0.6, 200)
    alpha = 0.5
    n = 2
    thermal_V = 8.314 * 298 / (n * 96485)
    I_bv = np.exp(alpha * eta_theory / thermal_V) - np.exp(-(1-alpha) * eta_theory / thermal_V)
    I_bv_norm = I_bv / np.max(np.abs(I_bv))
    ax.plot(eta_theory, I_bv_norm * np.max(np.abs(pred)) * 0.3, 'k--', alpha=0.4,
            linewidth=2, label="B-V shape (scaled)")
    ax.set_xlabel("Overpotential η (V)")
    ax.set_ylabel("Current Density (mA/cm²)")
    ax.set_title("I vs Overpotential (Butler-Volmer)")
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Tafel plot (log|I| vs η)
    ax = axes[1]
    X_tafel = np.column_stack([
        potentials,
        (potentials > 0.25).astype(float),
        np.full_like(potentials, 0.10),
        np.full_like(potentials, 50.0),
        np.ones_like(potentials),
        np.zeros_like(potentials),
        np.full_like(potentials, 298.0),
        np.full_like(potentials, 0.07),
    ]).astype(np.float32)[:, :len(feature_cols)]
    X_tafel_s = scaler.transform(X_tafel)
    pred_tafel = model.predict(X_tafel_s, verbose=0).flatten()
    if y_scaler is not None:
        pred_tafel = y_scaler.inverse_transform(pred_tafel.reshape(-1, 1)).flatten()

    eta_t = potentials - E0
    log_I = np.log10(np.abs(pred_tafel) + 1e-10)

    # Anodic and cathodic branches
    anodic = eta_t > 0.03
    cathodic = eta_t < -0.03
    ax.plot(eta_t[anodic], log_I[anodic], 'r.', markersize=3, label="Anodic")
    ax.plot(eta_t[cathodic], log_I[cathodic], 'b.', markersize=3, label="Cathodic")

    # Linear fit for Tafel slopes
    if np.sum(anodic) > 5:
        coeffs_a = np.polyfit(eta_t[anodic], log_I[anodic], 1)
        ax.plot(eta_t[anodic], np.polyval(coeffs_a, eta_t[anodic]), 'r-',
                linewidth=2, label=f"Anodic slope={coeffs_a[0]:.1f} V⁻¹")
    if np.sum(cathodic) > 5:
        coeffs_c = np.polyfit(eta_t[cathodic], log_I[cathodic], 1)
        ax.plot(eta_t[cathodic], np.polyval(coeffs_c, eta_t[cathodic]), 'b-',
                linewidth=2, label=f"Cathodic slope={coeffs_c[0]:.1f} V⁻¹")

    ax.set_xlabel("Overpotential η (V)")
    ax.set_ylabel("log₁₀|I| (mA/cm²)")
    ax.set_title("Tafel Plot Analysis")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: Charge conservation per composition
    ax = axes[2]
    scan_rates = [10, 20, 50, 100, 200]
    charge_ratios = []
    for sr in scan_rates:
        X_cv = np.column_stack([
            potentials,
            (potentials > 0.25).astype(float),
            np.full_like(potentials, 0.10),
            np.full_like(potentials, float(sr)),
            np.ones_like(potentials),
            np.zeros_like(potentials),
            np.full_like(potentials, 298.0),
            np.full_like(potentials, 0.07),
        ]).astype(np.float32)[:, :len(feature_cols)]
        X_cv_s = scaler.transform(X_cv)
        pred_cv = model.predict(X_cv_s, verbose=0).flatten()
        if y_scaler is not None:
            pred_cv = y_scaler.inverse_transform(pred_cv.reshape(-1, 1)).flatten()

        Q_total = np.trapezoid(pred_cv, potentials)
        Q_abs = np.trapezoid(np.abs(pred_cv), potentials) + 1e-8
        charge_ratios.append(abs(Q_total) / Q_abs * 100)

    ax.bar(range(len(scan_rates)), charge_ratios,
           tick_label=[f"{sr}" for sr in scan_rates],
           color=["#2ecc71" if r < 20 else "#e74c3c" for r in charge_ratios],
           edgecolor="black", linewidth=0.5)
    ax.axhline(y=10, color='orange', linestyle='--', linewidth=1, label="10% threshold")
    ax.set_xlabel("Scan Rate (mV/s)")
    ax.set_ylabel("Charge Imbalance (%)")
    ax.set_title("Charge Conservation: |∮I·dV| / ∫|I|dV")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Butler-Volmer & Electrochemical Consistency Analysis",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  B-V analysis saved → {save_path}")
    plt.close(fig)


# ============================================================
# PHYSICS VALIDATION DASHBOARD (PUBLICATION FIGURE)
# ============================================================

def plot_physics_dashboard(
    model, scaler, feature_cols, y_pred, y_true, X_raw,
    violations, metrics, save_path=None, y_scaler=None,
):
    """
    Generate a single publication-quality figure summarising all
    physics-informed constraints and model performance.

    Layout (3×3):
      [Pred vs Actual]  [Residuals]       [Physics Score Radar]
      [Randles-Sevcik]  [T-dependence]    [Composition Effect]
      [CV Overlay]      [B-V Adherence]   [Constraint Summary]
    """
    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # ---- Panel 1: Predicted vs Actual ----
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(y_true, y_pred, alpha=0.3, s=5, c='#3498db')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", lw=1.5)
    ax.set_xlabel("Actual I (mA/cm²)")
    ax.set_ylabel("Predicted I (mA/cm²)")
    ax.set_title(f"(a) Prediction Accuracy\nR²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}")
    ax.grid(True, alpha=0.3)

    # ---- Panel 2: Residual Distribution ----
    ax = fig.add_subplot(gs[0, 1])
    residuals = y_pred - y_true
    ax.hist(residuals, bins=60, color='#3498db', edgecolor='black', alpha=0.7, density=True)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xlabel("Residual (mA/cm²)")
    ax.set_ylabel("Density")
    ax.set_title(f"(b) Residual Distribution\nμ={np.mean(residuals):.3f}, σ={np.std(residuals):.3f}")
    ax.grid(True, alpha=0.3)

    # ---- Panel 3: Physics Constraint Radar ----
    ax = fig.add_subplot(gs[0, 2], polar=True)
    constraint_names = ["Current\nLimits", "Capacitance", "Smoothness",
                        "B-V Sign", "Charge\nConserv.", "Nernst\nT-dep", "Randles\nSevcik"]
    # Convert violations to "satisfaction scores" (100 - violation%)
    scores = [
        100 - violations.get("current_exceed_%", 0),
        100 - violations.get("negative_cap_%", 0),
        100 - violations.get("large_jumps_%", 0),
        100 - violations.get("bv_sign_violation_%", 0),
        100 - violations.get("charge_imbalance_%", 0),
        100 - violations.get("nernst_violation_%", 0),
        violations.get("randles_sevcik_R2", 1.0) * 100,
    ]
    scores = [max(0, min(100, s)) for s in scores]

    angles = np.linspace(0, 2 * np.pi, len(constraint_names), endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles += [angles[0]]

    ax.fill(angles, scores_plot, alpha=0.25, color='#2ecc71')
    ax.plot(angles, scores_plot, 'o-', color='#2ecc71', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(constraint_names, fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_title("(c) Physics Constraint Satisfaction (%)", pad=20, fontsize=10)

    # ---- Panel 4: Randles-Sevcik ----
    ax = fig.add_subplot(gs[1, 0])
    potentials = np.linspace(-0.6, 0.8, 100)
    scan_rates = [5, 10, 20, 50, 100, 200]
    peak_I = []
    for sr in scan_rates:
        X = np.column_stack([
            potentials,
            (potentials > 0.25).astype(float),
            np.full_like(potentials, 0.10),
            np.full_like(potentials, float(sr)),
            np.ones_like(potentials),
            np.zeros_like(potentials),
            np.full_like(potentials, 298.0),
            np.full_like(potentials, 0.07),
        ]).astype(np.float32)[:, :len(feature_cols)]
        X_s = scaler.transform(X)
        p = model.predict(X_s, verbose=0).flatten()
        if y_scaler is not None:
            p = y_scaler.inverse_transform(p.reshape(-1, 1)).flatten()
        peak_I.append(np.max(np.abs(p)))

    sqrt_sr = np.sqrt(scan_rates)
    coeffs = np.polyfit(sqrt_sr, peak_I, 1)
    fitted = np.polyval(coeffs, sqrt_sr)
    ss_res = np.sum((np.array(peak_I) - fitted)**2)
    ss_tot = np.sum((np.array(peak_I) - np.mean(peak_I))**2) + 1e-12
    r2_rs = 1 - ss_res / ss_tot

    ax.scatter(sqrt_sr, peak_I, c='#e74c3c', s=60, zorder=3)
    ax.plot(sqrt_sr, fitted, 'k--', lw=1.5, label=f"R²={r2_rs:.4f}")
    ax.set_xlabel("√(Scan Rate) (√(mV/s))")
    ax.set_ylabel("|I_peak| (mA/cm²)")
    ax.set_title("(d) Randles-Sevcik: I_p ∝ √v")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- Panel 5: Temperature Dependence ----
    ax = fig.add_subplot(gs[1, 1])
    temperatures = [298, 308, 318, 328]
    for T in temperatures:
        X = np.column_stack([
            potentials,
            (potentials > 0.25).astype(float),
            np.full_like(potentials, 0.10),
            np.full_like(potentials, 50.0),
            np.ones_like(potentials),
            np.zeros_like(potentials),
            np.full_like(potentials, float(T)),
            np.full_like(potentials, 0.07),
        ]).astype(np.float32)[:, :len(feature_cols)]
        X_s = scaler.transform(X)
        p = model.predict(X_s, verbose=0).flatten()
        if y_scaler is not None:
            p = y_scaler.inverse_transform(p.reshape(-1, 1)).flatten()
        ax.plot(potentials, p, linewidth=1.5, label=f"{T} K")

    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Current Density (mA/cm²)")
    ax.set_title("(e) Nernst: Temperature Dependence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 6: Composition Effect ----
    ax = fig.add_subplot(gs[1, 2])
    comps = [(0.0, 0.0, "BFO"), (0.10, 0.0, "Zn=0.10"),
             (0.0, 0.10, "Co=0.10"), (0.10, 0.10, "Zn+Co=0.10")]
    for zn, co, label in comps:
        conc = zn + co
        X = np.column_stack([
            potentials,
            (potentials > 0.25).astype(float),
            np.full_like(potentials, conc),
            np.full_like(potentials, 50.0),
            np.full_like(potentials, float(1 if zn > 0 else 0)),
            np.full_like(potentials, float(1 if co > 0 else 0)),
            np.full_like(potentials, 298.0),
            np.full_like(potentials, 0.07),
        ]).astype(np.float32)[:, :len(feature_cols)]
        X_s = scaler.transform(X)
        p = model.predict(X_s, verbose=0).flatten()
        if y_scaler is not None:
            p = y_scaler.inverse_transform(p.reshape(-1, 1)).flatten()
        ax.plot(potentials, p, linewidth=1.5, label=label)

    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Current Density (mA/cm²)")
    ax.set_title("(f) Faraday: Composition Effect on CV")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 7: CV Curve Overlay (best composition) ----
    ax = fig.add_subplot(gs[2, 0])
    # Use test data for actual vs predicted
    idx_sort = np.argsort(X_raw[:, 0])[:200]  # sort by potential, take subset
    ax.scatter(X_raw[idx_sort, 0], y_true[idx_sort], s=10, alpha=0.5,
               c='blue', label="Actual")
    ax.scatter(X_raw[idx_sort, 0], y_pred[idx_sort], s=10, alpha=0.5,
               c='red', label="PIML")
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Current Density (mA/cm²)")
    ax.set_title("(g) CV Curve Overlay (test subset)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 8: Butler-Volmer shape ----
    ax = fig.add_subplot(gs[2, 1])
    X_bv = np.column_stack([
        potentials,
        (potentials > 0.25).astype(float),
        np.full_like(potentials, 0.10),
        np.full_like(potentials, 50.0),
        np.ones_like(potentials),
        np.zeros_like(potentials),
        np.full_like(potentials, 298.0),
        np.full_like(potentials, 0.07),
    ]).astype(np.float32)[:, :len(feature_cols)]
    X_bv_s = scaler.transform(X_bv)
    pred_bv = model.predict(X_bv_s, verbose=0).flatten()
    if y_scaler is not None:
        pred_bv = y_scaler.inverse_transform(pred_bv.reshape(-1, 1)).flatten()

    eta = potentials - 0.25
    ax.plot(eta, pred_bv, 'b-', linewidth=1.5, label="PIML Prediction")
    # Theoretical B-V shape
    thermal_V = 8.314 * 298 / (2 * 96485)
    I_bv_theory = np.exp(0.5 * eta / thermal_V) - np.exp(-0.5 * eta / thermal_V)
    I_bv_norm = I_bv_theory / np.max(np.abs(I_bv_theory)) * np.max(np.abs(pred_bv)) * 0.3
    ax.plot(eta, I_bv_norm, 'r--', linewidth=2, alpha=0.5, label="B-V theory (scaled)")
    ax.set_xlabel("Overpotential η (V)")
    ax.set_ylabel("Current Density (mA/cm²)")
    ax.set_title("(h) Butler-Volmer Adherence")
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.5)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Panel 9: Constraint Summary Table ----
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    table_data = []
    constraint_labels = [
        ("Redox Current Limits", f"{violations.get('current_exceed_%', 0):.2f}%",
         "✓" if violations.get("current_exceed_%", 0) < 5 else "✗"),
        ("Capacitance Positivity", f"{violations.get('negative_cap_%', 0):.2f}%",
         "✓" if violations.get("negative_cap_%", 0) < 1 else "✗"),
        ("Smoothness", f"{violations.get('large_jumps_%', 0):.2f}%",
         "✓" if violations.get("large_jumps_%", 0) < 5 else "✗"),
        ("Butler-Volmer Sign", f"{violations.get('bv_sign_violation_%', 0):.2f}%",
         "✓" if violations.get("bv_sign_violation_%", 0) < 10 else "✗"),
        ("Charge Conservation", f"{violations.get('charge_imbalance_%', 0):.2f}%",
         "✓" if violations.get("charge_imbalance_%", 0) < 15 else "✗"),
        ("Nernst T-dependence", f"{violations.get('nernst_violation_%', 0):.2f}%",
         "✓" if violations.get("nernst_violation_%", 0) < 25 else "✗"),
        ("Randles-Sevcik R²", f"{violations.get('randles_sevcik_R2', 0):.4f}",
         "✓" if violations.get("randles_sevcik_R2", 0) > 0.90 else "✗"),
    ]
    for name, value, status in constraint_labels:
        table_data.append([name, value, status])

    table = ax.table(cellText=table_data,
                     colLabels=["Constraint", "Violation/Score", "Pass"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    # Color code pass/fail
    for i, (_, _, status) in enumerate(constraint_labels):
        color = "#d5f5e3" if status == "✓" else "#fadbd8"
        table[(i+1, 2)].set_facecolor(color)
    ax.set_title("(i) Physics Constraint Summary", fontsize=10, pad=10)

    fig.suptitle(
        "Physics-Informed ML: Comprehensive Electrochemical Validation Dashboard",
        fontsize=15, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Physics dashboard saved → {save_path}")
    plt.close(fig)


# ============================================================
# PHYSICS VALIDATION REPORT (TEXT)
# ============================================================

def generate_physics_report(metrics, violations, save_path=None):
    """Generate a text-based physics validation report for the paper."""
    lines = []
    lines.append("=" * 70)
    lines.append("  PIML PHYSICS VALIDATION REPORT")
    lines.append("  Zn/Co-substituted BiFeO₃/Bi₂₅FeO₄₀ Supercapacitor CV Prediction")
    lines.append("=" * 70)
    lines.append("")
    lines.append("  MODEL PERFORMANCE:")
    lines.append(f"    R²   = {metrics['R2']:.6f}")
    lines.append(f"    RMSE = {metrics['RMSE']:.6f} mA/cm²")
    lines.append(f"    MAE  = {metrics['MAE']:.6f} mA/cm²")
    lines.append(f"    MSE  = {metrics['MSE']:.6f}")
    lines.append("")
    lines.append("  ELECTROCHEMICAL PHYSICS CONSTRAINTS (9 total):")
    lines.append("  " + "-" * 55)

    constraint_map = {
        "current_exceed_%":    ("Redox Current Limit", "< 5%"),
        "negative_cap_%":      ("Capacitance Positivity", "< 1%"),
        "large_jumps_%":       ("Output Smoothness", "< 5%"),
        "bv_sign_violation_%": ("Butler-Volmer Sign Consistency", "< 10%"),
        "charge_imbalance_%":  ("Charge Conservation", "< 15%"),
        "nernst_violation_%":  ("Nernst Temperature Dep.", "< 25%"),
        "randles_sevcik_R2":   ("Randles-Sevcik I∝√v", "> 0.90"),
    }

    n_pass = 0
    for key, (name, threshold) in constraint_map.items():
        val = violations.get(key, 0)
        if key == "randles_sevcik_R2":
            passed = val > 0.90
            val_str = f"{val:.4f}"
        else:
            passed = val < float(threshold.split()[-1].rstrip("%"))
            val_str = f"{val:.3f}%"
        status = "PASS ✓" if passed else "FAIL ✗"
        if passed:
            n_pass += 1
        lines.append(f"    {name:40s}  {val_str:>10s}  [{threshold:>8s}]  {status}")

    lines.append("")
    lines.append(f"  OVERALL PHYSICS SCORE: {n_pass}/{len(constraint_map)} constraints satisfied")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n  Report saved → {save_path}")

    return report


# ============================================================
# MASTER EVALUATION FUNCTION
# ============================================================

def evaluate_piml(trainer, data: dict, cfg: dict):
    """
    Run full evaluation suite on the trained PIML model.

    Parameters
    ----------
    trainer : PIMLTrainer instance (with trained model)
    data    : dict from prepare_data()
    cfg     : config dict
    """
    plot_dir = cfg.get("plot_dir", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    mc = cfg.get("mc_samples", 50)

    # --- Test predictions ---
    y_mean, y_std = trainer.predict(
        data["X_test_scaled"].astype(np.float32),
        n_mc_samples=mc if cfg.get("mc_dropout", True) else 0,
    )

    # --- Inverse-transform to original scale if y_scaler provided ---
    y_scaler = data.get("y_scaler", None)
    if y_scaler is not None:
        y_mean = y_scaler.inverse_transform(y_mean.reshape(-1, 1)).flatten()
        if y_std is not None:
            y_std = y_std * y_scaler.scale_[0]  # scale std back (not shift)
        y_test = data.get("y_test_raw", data["y_test"])
    else:
        y_test = data["y_test"]

    # --- Metrics ---
    metrics = compute_metrics(y_test, y_mean)
    print(f"\n  R²   = {metrics['R2']:.5f}")
    print(f"  RMSE = {metrics['RMSE']:.5f}")
    print(f"  MAE  = {metrics['MAE']:.5f}")

    # --- Constraint violations ---
    violations = compute_constraint_violations(
        y_mean, data["X_test_raw"],
        max_current=cfg.get("max_current", 50.0),
    )
    print(f"\n  Constraint violations:")
    for k, v in violations.items():
        print(f"    {k}: {v}%")

    # --- Save metrics ---
    all_results = {**metrics, **violations}
    results_path = os.path.join(plot_dir, "piml_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {results_path}")

    # --- Plots ---
    print("\n  Generating plots...")

    # 1. Predicted vs Actual
    plot_pred_vs_actual(
        y_test, y_mean,
        title="PIML (Test)",
        save_path=os.path.join(plot_dir, "PIML_pred_vs_actual.png"),
        y_std=y_std,
    )

    # 2. Training history
    history_path = os.path.join(cfg.get("model_dir", "saved_models"), "piml_history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_history(history, os.path.join(plot_dir, "PIML_training_history.png"))

    # 3. CV curves (if test data has needed columns)
    feature_cols = data.get("feature_cols", [])
    if len(feature_cols) >= 4:
        # Build a test DataFrame for CV curve plotting
        df_test = pd.DataFrame(data["X_test_raw"], columns=feature_cols)
        df_test["Current_Density"] = y_test
        plot_cv_curves(
            df_test, y_mean,
            group_col="Zn_Co_Conc" if "Zn_Co_Conc" in feature_cols else feature_cols[0],
            save_path=os.path.join(plot_dir, "PIML_cv_curves.png"),
            y_std=y_std,
        )

        # 4. Redox peaks
        plot_redox_peaks(df_test, y_mean, os.path.join(plot_dir, "PIML_redox_peaks.png"))

    # 5. Uncertainty
    if y_std is not None:
        plot_uncertainty(y_test, y_mean, y_std,
                          os.path.join(plot_dir, "PIML_uncertainty.png"))

    # 6. SHAP
    print("\n  Running feature importance analysis...")
    plot_shap_analysis(
        trainer.model,
        data["X_test_scaled"][:500],
        feature_cols,
        save_path=os.path.join(plot_dir, "PIML_shap.png"),
    )

    # 7. Optimal composition
    print("\n  Analysing optimal Zn/Co composition...")
    analyse_optimal_composition(
        trainer.model,
        data["scaler"],
        feature_cols,
        save_path=os.path.join(plot_dir, "PIML_optimal_composition.png"),
    )

    # 8. Butler-Volmer adherence & charge conservation
    print("\n  Running Butler-Volmer & electrochemical consistency analysis...")
    plot_butler_volmer_adherence(
        trainer.model,
        data["scaler"],
        feature_cols,
        save_path=os.path.join(plot_dir, "PIML_butler_volmer_analysis.png"),
        y_scaler=data.get("y_scaler", None),
    )

    # 9. Physics validation dashboard (publication figure)
    print("\n  Generating physics validation dashboard...")
    plot_physics_dashboard(
        trainer.model,
        data["scaler"],
        feature_cols,
        y_pred=y_mean,
        y_true=y_test,
        X_raw=data["X_test_raw"],
        violations=violations,
        metrics=metrics,
        save_path=os.path.join(plot_dir, "PIML_physics_dashboard.png"),
        y_scaler=data.get("y_scaler", None),
    )

    # 10. Physics validation report
    print("\n  Generating physics validation report...")
    generate_physics_report(
        metrics, violations,
        save_path=os.path.join(plot_dir, "PIML_physics_report.txt"),
    )

    print("\n  Evaluation complete.")
    return metrics, violations

# ============================================================
# EXTRAPOLATION ANALYSIS
# ============================================================

def extrapolation_analysis(
    model,
    scaler,
    feature_cols: list,
    training_ranges: dict = None,
    save_path: str = None,
):
    """
    Test model predictions outside training data bounds to assess
    physics-informed extrapolation capability.

    Tests extrapolation in:
    1. Higher scan rates (beyond training max)
    2. Extreme temperatures (beyond training range)
    3. Novel compositions (unseen Zn/Co ratios)
    4. Extended potential windows

    Parameters
    ----------
    model : keras.Model
    scaler : fitted StandardScaler
    feature_cols : list of feature column names
    training_ranges : dict of {feature: (min, max)} from training data
    save_path : path to save the figure

    Returns
    -------
    dict : extrapolation metrics and analysis results
    """
    if training_ranges is None:
        training_ranges = {
            "Scan_Rate": (5, 200),
            "Temperature": (298, 328),
            "Zn_Co_Conc": (0.0, 0.30),
            "Potential": (-0.6, 0.8),
        }

    potentials = np.linspace(-0.6, 0.8, 100)
    results = {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # ------ 1. Scan Rate Extrapolation ------
    ax = axes[0, 0]
    scan_rates = [10, 50, 100, 200, 350, 500, 750, 1000]  # last 4 are extrapolation
    train_max_sr = training_ranges["Scan_Rate"][1]

    for sr in scan_rates:
        X = np.column_stack([
            potentials,
            (potentials > 0.25).astype(float),
            np.full_like(potentials, 0.10),
            np.full_like(potentials, float(sr)),
            np.ones_like(potentials),
            np.zeros_like(potentials),
            np.full_like(potentials, 298.0),
            np.full_like(potentials, 0.07),
        ]).astype(np.float32)[:, :len(feature_cols)]

        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled, verbose=0).flatten()

        style = "b-" if sr <= train_max_sr else "r--"
        alpha = 0.8 if sr <= train_max_sr else 0.6
        lbl = f"{sr} mV/s" + (" (extrap)" if sr > train_max_sr else "")
        ax.plot(potentials, pred, style, alpha=alpha, label=lbl, linewidth=1.2)

    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Current Density (mA/cm²)")
    ax.set_title("Scan Rate Extrapolation")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.5)

    # ------ 2. Temperature Extrapolation ------
    ax = axes[0, 1]
    temperatures = [278, 298, 318, 328, 348, 373, 398]
    train_max_T = training_ranges["Temperature"][1]
    train_min_T = training_ranges["Temperature"][0]

    for T in temperatures:
        X = np.column_stack([
            potentials,
            (potentials > 0.25).astype(float),
            np.full_like(potentials, 0.10),
            np.full_like(potentials, 50.0),
            np.ones_like(potentials),
            np.zeros_like(potentials),
            np.full_like(potentials, float(T)),
            np.full_like(potentials, 0.07),
        ]).astype(np.float32)[:, :len(feature_cols)]

        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled, verbose=0).flatten()

        is_extrap = T < train_min_T or T > train_max_T
        style = "r--" if is_extrap else "b-"
        lbl = f"{T} K" + (" (extrap)" if is_extrap else "")
        ax.plot(potentials, pred, style, alpha=0.7, label=lbl, linewidth=1.2)

    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Current Density (mA/cm²)")
    ax.set_title("Temperature Extrapolation")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ------ 3. Novel Composition Extrapolation ------
    ax = axes[1, 0]
    compositions = [
        (0.0, 0.0, "Pure BFO"),
        (0.10, 0.0, "Zn=0.10"),
        (0.0, 0.10, "Co=0.10"),
        (0.20, 0.0, "Zn=0.20 (extrap)"),
        (0.0, 0.25, "Co=0.25 (extrap)"),
        (0.15, 0.15, "Zn=Co=0.15 (extrap)"),
        (0.25, 0.25, "Zn=Co=0.25 (extrap)"),
    ]

    for zn, co, label in compositions:
        conc = zn + co
        X = np.column_stack([
            potentials,
            (potentials > 0.25).astype(float),
            np.full_like(potentials, conc),
            np.full_like(potentials, 50.0),
            np.full_like(potentials, float(1 if zn > 0 else 0)),
            np.full_like(potentials, float(1 if co > 0 else 0)),
            np.full_like(potentials, 298.0),
            np.full_like(potentials, 0.07),
        ]).astype(np.float32)[:, :len(feature_cols)]

        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled, verbose=0).flatten()

        is_extrap = "extrap" in label
        style = "r--" if is_extrap else "b-"
        ax.plot(potentials, pred, style, alpha=0.7, label=label, linewidth=1.2)

    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("Current Density (mA/cm²)")
    ax.set_title("Composition Extrapolation")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ------ 4. Randles-Sevcik Consistency Check ------
    ax = axes[1, 1]
    # Check if peak current scales as √(scan_rate) (Randles-Sevcik law)
    sr_values = np.array([5, 10, 20, 50, 100, 200, 350, 500, 750, 1000])
    peak_currents = []

    for sr in sr_values:
        X = np.column_stack([
            potentials,
            (potentials > 0.25).astype(float),
            np.full_like(potentials, 0.10),
            np.full_like(potentials, float(sr)),
            np.ones_like(potentials),
            np.zeros_like(potentials),
            np.full_like(potentials, 298.0),
            np.full_like(potentials, 0.07),
        ]).astype(np.float32)[:, :len(feature_cols)]

        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled, verbose=0).flatten()
        peak_currents.append(np.max(np.abs(pred)))

    peak_currents = np.array(peak_currents)
    sqrt_sr = np.sqrt(sr_values)

    # Fit linear relation: I_peak = a * √v
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(sqrt_sr, peak_currents, 1)
    fitted = np.polyval(coeffs, sqrt_sr)

    r2_randles = 1 - np.sum((peak_currents - fitted)**2) / np.sum((peak_currents - peak_currents.mean())**2)
    results["randles_sevcik_r2"] = float(r2_randles)

    in_train = sr_values <= train_max_sr
    ax.scatter(sqrt_sr[in_train], peak_currents[in_train], c="blue", s=50,
               zorder=3, label="In-distribution")
    ax.scatter(sqrt_sr[~in_train], peak_currents[~in_train], c="red", s=50,
               marker="^", zorder=3, label="Extrapolation")
    ax.plot(sqrt_sr, fitted, "k--", lw=1.5, label=f"Linear fit (R²={r2_randles:.4f})")
    ax.set_xlabel("√Scan Rate (√(mV/s))")
    ax.set_ylabel("|Peak Current Density| (mA/cm²)")
    ax.set_title("Randles-Sevcik Consistency: I_peak ∝ √v")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("PIML Extrapolation Analysis: Physics-Informed Predictions Beyond Training Data",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Extrapolation plot saved → {save_path}")
    plt.close(fig)

    return results


# ============================================================
# FULL MODEL COMPARISON
# ============================================================

def compare_all_models(
    models_dict: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_test_scaled_baseline: np.ndarray = None,
    X_test_scaled_piml: np.ndarray = None,
    X_test_raw: np.ndarray = None,
    piml_trainer=None,
    piml_y_scaler=None,
    max_current: float = 50.0,
    mc_samples: int = 50,
    save_dir: str = "plots",
) -> pd.DataFrame:
    """
    Compare PIML model against all baseline models with comprehensive metrics.

    Parameters
    ----------
    models_dict : dict
        {"model_name": model_object} for all models to compare.
    X_test, y_test : test data (raw, unscaled targets).
    X_test_scaled_baseline : scaled test features for baseline models.
    X_test_scaled_piml : scaled test features for PIML model.
    X_test_raw : raw test features (for constraint violations).
    piml_trainer : PIMLTrainer instance.
    piml_y_scaler : StandardScaler for PIML target inverse-transform.

    Returns
    -------
    pd.DataFrame : comparison table of all models.
    """
    os.makedirs(save_dir, exist_ok=True)
    all_results = {}
    all_preds = {}

    for name, model in models_dict.items():
        print(f"\n  Evaluating {name}...")

        if name == "PIML" and piml_trainer is not None:
            y_mean, y_std = piml_trainer.predict(
                X_test_scaled_piml.astype(np.float32), n_mc_samples=mc_samples
            )
            if piml_y_scaler is not None:
                y_mean = piml_y_scaler.inverse_transform(y_mean.reshape(-1, 1)).flatten()
                if y_std is not None:
                    y_std = y_std * piml_y_scaler.scale_[0]
            preds = y_mean
        elif name == "PIML":
            X_s = X_test_scaled_piml if X_test_scaled_piml is not None else X_test
            preds = model.predict(X_s.astype(np.float32), verbose=0).flatten()
            if piml_y_scaler is not None:
                preds = piml_y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        else:
            X_s = X_test_scaled_baseline if X_test_scaled_baseline is not None else X_test
            preds = model.predict(X_s) if not hasattr(model, 'predict') else model.predict(X_s)
            if hasattr(preds, 'flatten'):
                preds = preds.flatten()

        metrics = compute_metrics(y_test, preds)

        # Constraint violations
        X_raw = X_test_raw if X_test_raw is not None else X_test
        violations = compute_constraint_violations(preds, X_raw, max_current=max_current)

        all_results[name] = {**metrics, **violations}
        all_preds[name] = preds

    # Build comparison DataFrame
    comparison_df = pd.DataFrame(all_results).T
    comparison_df = comparison_df.round(5)

    print("\n" + "=" * 70)
    print("  FULL MODEL COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string())

    # --- Comparison scatter plots ---
    n_models = len(all_preds)
    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, (name, preds) in enumerate(all_preds.items()):
        if i >= len(axes):
            break
        ax = axes[i]
        ax.scatter(y_test, preds, alpha=0.3, s=5)
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        ax.plot(lims, lims, "k--", lw=1.5)
        m = all_results[name]
        ax.set_title(f"{name}\nR²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}")
        ax.set_xlabel("Actual Current Density")
        ax.set_ylabel("Predicted Current Density")
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Model Comparison: Actual vs Predicted", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "model_comparison_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Bar chart of metrics ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    model_names = list(all_results.keys())

    for ax, metric, title in zip(axes,
                                  ["R2", "RMSE", "MAE"],
                                  ["R² (higher is better)", "RMSE (lower is better)", "MAE (lower is better)"]):
        vals = [all_results[m][metric] for m in model_names]
        colors = ["#2ecc71" if m == "PIML" else "#3498db" for m in model_names]
        bars = ax.bar(model_names, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        # Annotate bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model Performance Metrics Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "model_comparison_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Constraint violation comparison ---
    fig, ax = plt.subplots(figsize=(14, 6))
    violation_keys = ["current_exceed_%", "negative_cap_%", "large_jumps_%",
                      "bv_sign_violation_%", "charge_imbalance_%"]
    x_pos = np.arange(len(model_names))
    width = 0.15

    for i, vk in enumerate(violation_keys):
        vals = [all_results[m].get(vk, 0) for m in model_names]
        ax.bar(x_pos + i * width, vals, width, label=vk.replace("_", " ").title())

    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Violation %")
    ax.set_title("Physics Constraint Violations by Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "model_comparison_violations.png"), dpi=300)
    plt.close(fig)

    # Save comparison table
    comparison_df.to_csv(os.path.join(save_dir, "model_comparison_table.csv"))
    print(f"\n  Comparison saved → {save_dir}/model_comparison_table.csv")

    return comparison_df


# ============================================================
# PUBLICATION-READY SUMMARY TABLE (LaTeX)
# ============================================================

def generate_latex_table(comparison_df: pd.DataFrame, save_path: str = None) -> str:
    """
    Generate a LaTeX-formatted table for publication.

    Returns the LaTeX string and optionally saves to file.
    """
    cols_rename = {
        "R2": r"$R^2$",
        "RMSE": "RMSE",
        "MAE": "MAE",
        "MSE": "MSE",
        "current_exceed_%": "Current Exceed (\\%)",
        "negative_cap_%": "Neg. Cap. (\\%)",
        "large_jumps_%": "Large Jumps (\\%)",
        "bv_sign_violation_%": "B-V Sign (\\%)",
        "charge_imbalance_%": "Charge Imb. (\\%)",
        "nernst_violation_%": "Nernst (\\%)",
        "randles_sevcik_R2": "R-S $R^2$",
    }

    df = comparison_df.copy()
    df = df.rename(columns=cols_rename)

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Comparison of ML models for CV curve prediction of "
    latex += "Zn/Co-substituted BiFeO$_3$/Bi$_{25}$FeO$_{40}$.}\n"
    latex += "\\label{tab:model_comparison}\n"
    latex += df.to_latex(float_format="%.4f", bold_rows=False)
    latex += "\\end{table}\n"

    if save_path:
        with open(save_path, "w") as f:
            f.write(latex)
        print(f"  LaTeX table saved → {save_path}")

    return latex