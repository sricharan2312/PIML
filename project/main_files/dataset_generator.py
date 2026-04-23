"""
Dataset Generator for Physics-Informed ML Model
=================================================
Generates synthetic CV (Cyclic Voltammetry) data grounded in electrochemical physics
for Zn/Co-substituted BiFeO3/Bi25FeO40 materials.

Physics basis:
  - Randles-Sevcik equation for peak current dependence on scan rate
  - Nernst equation for redox peak potential shift with temperature
  - Butler-Volmer kinetics for current-overpotential relationship
  - Capacitive current contribution: I_cap = C_dl * v (double-layer charging)
  - Faradaic current from redox processes (Gaussian-shaped peaks)
  - Charge conservation: anodic ~ cathodic charge (reversible processes)
  - Thermodynamic consistency: dG = -nFE driving force

This module can also load real experimental data from the existing Excel dataset.
"""

import os
import numpy as np
import pandas as pd
from itertools import product


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================
FARADAY = 96485.3329       # C/mol
R_GAS   = 8.314            # J/(mol·K)
N_ELEC  = 2                # electrons transferred in redox couple
ALPHA   = 0.5              # Butler-Volmer transfer coefficient


# ============================================================
# HELPER: ELECTROCHEMICAL CURRENT MODEL
# ============================================================

def _gaussian_peak(potential, center, sigma, amplitude):
    """Gaussian-shaped faradaic peak."""
    return amplitude * np.exp(-0.5 * ((potential - center) / sigma) ** 2)


def _cv_current(potential, scan_rate, temperature, electrode_area,
                zn_frac, co_frac, e_ox, e_red, noise_std=0.0):
    """
    Compute physically-motivated CV current for a single sweep direction.

    Parameters
    ----------
    potential : ndarray   – voltage points (V)
    scan_rate : float     – mV/s
    temperature : float   – K
    electrode_area : float – cm²
    zn_frac, co_frac : float – molar substitution fractions (0-1)
    e_ox, e_red : float   – oxidation / reduction peak centres (V)
    noise_std : float     – Gaussian noise standard deviation on current
    """
    v = scan_rate * 1e-3  # mV/s → V/s

    # --- Capacitive baseline ---
    # C_dl varies with composition (higher doping → higher surface area → higher C_dl)
    C_dl = (15.0 + 40.0 * zn_frac + 35.0 * co_frac) * 1e-6  # µF → F
    I_cap = C_dl * electrode_area * v  # A

    # --- Faradaic peaks (Randles-Sevcik scaling) ---
    # ip ∝ n^(3/2) * A * D^(1/2) * C * v^(1/2)
    # We model D and C as composition-dependent

    D_eff = (1.0 + 0.5 * zn_frac + 0.3 * co_frac) * 1e-6   # cm²/s effective diffusion
    C_bulk = (5.0 + 2.0 * zn_frac + 3.0 * co_frac) * 1e-3   # mol/cm³

    randles_coeff = 0.4463 * (N_ELEC ** 1.5) * FARADAY * electrode_area \
                    * np.sqrt(D_eff * FARADAY / (R_GAS * temperature))
    ip_scale = randles_coeff * C_bulk * np.sqrt(v)  # Amps at peak

    # Peak width broadens slightly with scan rate & temperature
    sigma = 0.04 + 0.005 * np.log10(v + 1e-12) + 0.0001 * (temperature - 298)

    # Temperature shifts peak positions (Nernst-like)
    shift = (R_GAS * temperature / (N_ELEC * FARADAY)) * np.log(1 + 0.1 * (zn_frac + co_frac))
    e_ox_eff  = e_ox + shift
    e_red_eff = e_red - shift

    # Oxidation (positive) and reduction (negative) peaks
    I_ox  =  _gaussian_peak(potential, e_ox_eff,  sigma, ip_scale)
    I_red = -_gaussian_peak(potential, e_red_eff, sigma, ip_scale * 0.9)  # slightly asymmetric

    # --- Butler-Volmer contribution near equilibrium ---
    # Adds the characteristic exponential current-overpotential shape
    E0_mid = (e_ox_eff + e_red_eff) / 2.0
    eta = potential - E0_mid
    thermal_V = (R_GAS * temperature) / (N_ELEC * FARADAY)
    # Exchange current density scales with concentration and area
    I0 = (0.01 + 0.05 * (zn_frac + co_frac)) * electrode_area * 1e-3  # A
    # Butler-Volmer: I = I0 * [exp(alpha*eta/V_T) - exp(-(1-alpha)*eta/V_T)]
    # Clip exponents to ±8 (prevents exp overflow while keeping physical shape)
    exp_anod = np.exp(np.clip(ALPHA * eta / thermal_V, -8, 8))
    exp_cath = np.exp(np.clip(-(1 - ALPHA) * eta / thermal_V, -8, 8))
    I_bv = I0 * (exp_anod - exp_cath)
    # Attenuate far from equilibrium – tight Gaussian envelope
    # (BV is most accurate within ~±100 mV of E⁰; beyond that, mass-transport limits apply)
    bv_envelope = np.exp(-0.5 * (eta / 0.08) ** 2)
    I_bv *= bv_envelope
    # Cap BV contribution so it never exceeds the faradaic peak current
    I_bv = np.clip(I_bv, -ip_scale, ip_scale)

    I_total = I_cap + I_ox + I_red + I_bv  # A

    # Convert to mA/cm² (current density)
    j = (I_total / electrode_area) * 1e3  # mA/cm²

    if noise_std > 0:
        j += np.random.normal(0, noise_std, size=j.shape)
    return j, e_ox_eff, e_red_eff


# ============================================================
# FULL SYNTHETIC DATASET
# ============================================================

def generate_synthetic_dataset(
    n_potential_points: int = 200,
    potential_range: tuple = (-0.6, 0.8),
    scan_rates: list = None,
    temperatures: list = None,
    electrode_areas: list = None,
    compositions: list = None,
    noise_std: float = 0.02,
    seed: int = 42,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Generate a comprehensive synthetic CV dataset.

    Returns DataFrame with columns:
        Potential, Zn_frac, Co_frac, Zn_Co_Conc, Scan_Rate, Temperature,
        Electrode_Area, ZN (binary), CO (binary), OXIDATION,
        Current_Density, Specific_Capacitance, Redox_Peak_Ox, Redox_Peak_Red
    """
    rng = np.random.RandomState(seed)

    if scan_rates is None:
        scan_rates = [5, 10, 20, 50, 100, 200]  # mV/s
    if temperatures is None:
        temperatures = [298, 308, 318, 328]       # K
    if electrode_areas is None:
        electrode_areas = [0.07, 0.10, 0.15]       # cm²
    if compositions is None:
        compositions = [
            (0.0, 0.0),   # Pure BFO
            (0.05, 0.0),  # Low Zn
            (0.10, 0.0),  # Med Zn
            (0.15, 0.0),  # High Zn
            (0.0, 0.05),  # Low Co
            (0.0, 0.10),  # Med Co
            (0.0, 0.15),  # High Co
            (0.05, 0.05), # Mixed low
            (0.10, 0.10), # Mixed med
            (0.15, 0.05), # Mixed high Zn / low Co
            (0.05, 0.15), # Mixed low Zn / high Co
        ]

    # Base redox peak positions for BFO
    E_OX_BASE  = 0.35   # V
    E_RED_BASE = 0.15   # V

    potentials = np.linspace(potential_range[0], potential_range[1], n_potential_points)

    records = []
    for (zn, co), sr, T, A in product(compositions, scan_rates, temperatures, electrode_areas):
        j, e_ox, e_red = _cv_current(
            potentials, sr, T, A, zn, co,
            E_OX_BASE, E_RED_BASE,
            noise_std=noise_std * rng.uniform(0.5, 1.5)
        )

        # Compute specific capacitance using ∫I dV / (v * ΔV * m)
        # Approximate: C_sp = integral(|j|) * dV / (scan_rate_V * voltage_window)
        dV = potentials[1] - potentials[0]
        v_Vs = sr * 1e-3
        voltage_window = potential_range[1] - potential_range[0]
        C_sp = np.trapz(np.abs(j), potentials) / (v_Vs * voltage_window)

        zn_binary = 1 if zn > 0 else 0
        co_binary = 1 if co > 0 else 0
        conc = zn + co

        for i, V in enumerate(potentials):
            oxidation_flag = 1 if V >= (e_ox + e_red) / 2 else 0
            records.append({
                "Potential": round(V, 6),
                "Zn_frac": zn,
                "Co_frac": co,
                "Zn_Co_Conc": conc,
                "Scan_Rate": sr,
                "Temperature": T,
                "Electrode_Area": A,
                "ZN": zn_binary,
                "CO": co_binary,
                "OXIDATION": oxidation_flag,
                "Current_Density": round(j[i], 6),
                "Specific_Capacitance": round(C_sp, 4),
                "Redox_Peak_Ox": round(e_ox, 6),
                "Redox_Peak_Red": round(e_red, 6),
            })

    df = pd.DataFrame(records)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Synthetic dataset saved → {save_path}  ({len(df)} rows)")

    return df


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_dataset(
    df: pd.DataFrame,
    noise_factor: float = 0.02,
    n_augmented: int = 1,
    jitter_features: list = None,
    interpolation: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Augment a CV dataset using multiple physics-aware strategies.

    Strategies
    ----------
    1. **Gaussian noise injection**: Add small noise to current density and features.
    2. **Feature jitter**: Small perturbations to scan rate, temperature, composition.
    3. **Interpolation**: Create synthetic samples by interpolating between existing
       compositions (SMOTE-like for regression).
    4. **Scan-rate scaling**: Scale current using known I ∝ √v relation (Randles-Sevcik).

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset.
    noise_factor : float
        Standard deviation of Gaussian noise (relative to feature std).
    n_augmented : int
        Number of augmented copies per original sample.
    jitter_features : list
        Columns to apply feature jitter to.
    interpolation : bool
        If True, add interpolated samples between compositions.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Combined original + augmented dataset.
    """
    rng = np.random.RandomState(seed)
    frames = [df.copy()]

    if jitter_features is None:
        jitter_features = ["Temperature", "Electrode_Area", "Scan_Rate"]

    # ------ Strategy 1: Gaussian noise on current density ------
    for i in range(n_augmented):
        aug = df.copy()
        if "Current_Density" in aug.columns:
            std = aug["Current_Density"].std()
            aug["Current_Density"] += rng.normal(0, noise_factor * std, size=len(aug))
        # Add small noise to features as well
        for col in jitter_features:
            if col in aug.columns:
                col_std = aug[col].std()
                if col_std > 0:
                    aug[col] += rng.normal(0, noise_factor * col_std, size=len(aug))
        frames.append(aug)

    # ------ Strategy 2: Randles-Sevcik scan-rate scaling ------
    # Generate samples at new scan rates using I ∝ √v scaling
    if "Scan_Rate" in df.columns and "Current_Density" in df.columns:
        new_scan_rates = [7, 15, 30, 75, 150]  # intermediate scan rates
        existing_rates = sorted(df["Scan_Rate"].unique())
        for new_sr in new_scan_rates:
            if new_sr in existing_rates:
                continue
            # Find closest existing scan rate
            closest_sr = min(existing_rates, key=lambda x: abs(x - new_sr))
            sub = df[df["Scan_Rate"] == closest_sr].copy()
            # Scale current: I_new = I_old * √(v_new / v_old)
            scale_factor = np.sqrt(new_sr / closest_sr)
            sub["Current_Density"] *= scale_factor
            sub["Scan_Rate"] = new_sr
            # Update capacitance if present
            if "Specific_Capacitance" in sub.columns:
                # C_sp ∝ 1/v for capacitive, so approximate re-scaling
                sub["Specific_Capacitance"] *= (closest_sr / new_sr)
            # Add small noise
            sub["Current_Density"] += rng.normal(0, 0.01 * sub["Current_Density"].std(), len(sub))
            frames.append(sub)

    # ------ Strategy 3: Composition interpolation ------
    if interpolation and "Zn_frac" in df.columns and "Co_frac" in df.columns:
        comps = df[["Zn_frac", "Co_frac"]].drop_duplicates().values
        if len(comps) >= 2:
            # Pick random pairs and interpolate
            n_interp = min(len(comps) * 3, 20)
            for _ in range(n_interp):
                idx1, idx2 = rng.choice(len(comps), 2, replace=False)
                alpha = rng.uniform(0.2, 0.8)
                zn_new = alpha * comps[idx1, 0] + (1 - alpha) * comps[idx2, 0]
                co_new = alpha * comps[idx1, 1] + (1 - alpha) * comps[idx2, 1]

                # Get data for each composition
                mask1 = (df["Zn_frac"] == comps[idx1, 0]) & (df["Co_frac"] == comps[idx1, 1])
                mask2 = (df["Zn_frac"] == comps[idx2, 0]) & (df["Co_frac"] == comps[idx2, 1])
                df1 = df[mask1].copy().reset_index(drop=True)
                df2 = df[mask2].copy().reset_index(drop=True)

                # Match by potential and scan rate for interpolation
                if len(df1) > 0 and len(df2) > 0:
                    # Use the smaller subset size
                    n_rows = min(len(df1), len(df2))
                    interp = df1.iloc[:n_rows].copy()
                    interp["Zn_frac"] = zn_new
                    interp["Co_frac"] = co_new
                    interp["Zn_Co_Conc"] = zn_new + co_new
                    interp["ZN"] = 1 if zn_new > 0.001 else 0
                    interp["CO"] = 1 if co_new > 0.001 else 0
                    if "Current_Density" in interp.columns:
                        j1 = df1["Current_Density"].values[:n_rows]
                        j2 = df2["Current_Density"].values[:n_rows]
                        interp["Current_Density"] = alpha * j1 + (1 - alpha) * j2
                    frames.append(interp)

    augmented = pd.concat(frames, ignore_index=True)
    print(f"Augmentation: {len(df)} -> {len(augmented)} samples "
          f"(+{len(augmented) - len(df)} augmented)")
    return augmented


# ============================================================
# LOAD REAL EXPERIMENTAL DATA (EXISTING EXCEL)
# ============================================================

def load_experimental_data(path: str = None) -> pd.DataFrame:
    """
    Load the original experimental CV dataset and add derived columns
    needed by the PIML model.
    """
    if path is None:
        candidates = [
            "Dataset/Training Dataset/CV_DATASET (2).xlsx",
            os.path.join("project", "Dataset", "Training Dataset", "CV_DATASET (2).xlsx"),
        ]
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        if path is None:
            raise FileNotFoundError("Could not locate experimental dataset.")

    df = pd.read_excel(path)

    # ---- Map existing columns to unified schema ----
    rename_map = {
        "Potential": "Potential",
        "Current": "Current_Density",
        "OXIDATION": "OXIDATION",
        "Zn/Co_Conc": "Zn_Co_Conc",
        "SCAN_RATE": "Scan_Rate",
        "ZN": "ZN",
        "CO": "CO",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Add synthetic placeholders for columns the experimental data may lack
    if "Temperature" not in df.columns:
        df["Temperature"] = 298.0
    if "Electrode_Area" not in df.columns:
        df["Electrode_Area"] = 0.07
    if "Zn_frac" not in df.columns:
        df["Zn_frac"] = df["Zn_Co_Conc"] * df["ZN"]
    if "Co_frac" not in df.columns:
        df["Co_frac"] = df["Zn_Co_Conc"] * df["CO"]

    return df


# ============================================================
# CLI ENTRY-POINT
# ============================================================

if __name__ == "__main__":
    print("Generating synthetic CV dataset for PIML training...")
    df = generate_synthetic_dataset(
        n_potential_points=200,
        noise_std=0.03,
        save_path="Dataset/synthetic_cv_dataset.csv",
    )
    print(f"\nDataset shape: {df.shape}")
    print(df.head(10))
    print(f"\nFeature ranges:")
    for col in df.columns:
        print(f"  {col:30s}  min={df[col].min():.4f}  max={df[col].max():.4f}")
