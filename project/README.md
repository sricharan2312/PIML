# Physics-Informed Machine Learning for CV Curve Prediction

## Zn/Co-substituted BiFeO₃/Bi₂₅FeO₄₀ Electrochemical Performance

A **physics-informed hybrid machine learning** framework for predicting cyclic voltammetry (CV) curves and electrochemical performance of Zn/Co-substituted BiFeO₃/Bi₂₅FeO₄₀ materials for supercapacitor applications.

### Supporting Information

- [Machine Learning-Based Prediction of Cyclic Voltammetry Behavior of Substitution of Zinc and Cobalt in BiFeO3/Bi25FeO40 for Supercapacitor Applications](https://pubs.acs.org/doi/full/10.1021/acsomega.3c10485)

---

## Novelty

1. **Physics-Informed Loss Function (9 Constraints)** — Embeds Faraday’s law, Butler-Volmer kinetics, Nernst equation, capacitance relation, redox limits, charge conservation, Randles-Sevcik scaling, thermodynamic consistency, and smoothness constraints directly into neural network training
2. **Hybrid Model** — Combines data-driven MSE with 9 physics-based penalty terms:
   $$\mathcal{L}_{total} = \text{MSE}_{data} + \sum_{i=1}^{9} \lambda_i \mathcal{L}_{physics,i}$$
3. **Butler-Volmer Kinetics** — Enforces the fundamental electrode kinetics equation: $I = I_0[e^{\alpha n F \eta / RT} - e^{-(1-\alpha) n F \eta / RT}]$
4. **Nernst Equation Compliance** — Temperature dependence follows $E = E^0 + (RT/nF)\ln(C_{ox}/C_{red})$
5. **Charge Conservation** — Net charge over a full CV cycle enforced to be approximately zero: $\oint I \cdot dV \approx 0$
6. **Uncertainty Quantification** — MC-Dropout (Bayesian approximation) provides calibrated 95% prediction intervals
7. **Composition Optimisation** — Automated search for optimal Zn/Co substitution ratios maximising capacitance
8. **Improved Extrapolation** — Physics constraints enforce Randles-Sevcik scaling ($I_{peak} \propto \sqrt{v}$) even outside training data
9. **Data Augmentation** — Physics-aware augmentation strategies (scan-rate scaling, composition interpolation, noise injection)
10. **Comprehensive Comparison** — PIML benchmarked against ANN, Random Forest, XGBoost, and Meta-model baselines
11. **Publication-Quality Dashboard** — 9-panel figure with radar chart, Tafel plot, and constraint summary table

---

## Project Structure

```
project/
├── PIML_notebook.ipynb              # Reproducible Jupyter notebook (full pipeline)
├── README.md                        # This file
├── requirements.txt
├── Dataset/
│   ├── Training Dataset/
│   │   └── CV_DATASET (2).xlsx      # Experimental data
│   ├── Testing Dataset/
│   └── synthetic_cv_dataset.csv     # Generated synthetic data
├── main_files/
│   ├── PIML_model.py                # Physics-informed NN architectures & loss functions
│   ├── PIML_training.py             # End-to-end training pipeline
│   ├── PIML_evaluation.py           # Evaluation, plots, SHAP, extrapolation analysis
│   ├── dataset_generator.py         # Synthetic CV data generator + augmentation
│   ├── unified_model_interface.py   # CLI interface for all models (incl. PIML)
│   ├── ANN_code.py                  # Baseline ANN
│   ├── RF_code.py                   # Baseline Random Forest
│   ├── XGB_code.py                  # Baseline XGBoost
│   └── Meta-model_code.py           # Baseline Stacking Ensemble
├── saved_models/
│   ├── piml_model.keras             # Trained PIML weights
│   ├── piml_scaler.pkl              # Feature scaler
│   ├── piml_y_scaler.pkl            # Target scaler
│   ├── piml_history.json            # Training loss history
│   └── piml_config.json             # Model configuration
└── plots/
    ├── PIML_pred_vs_actual.png      # Predicted vs actual scatter
    ├── PIML_cv_curves.png           # CV curve overlay (experimental vs predicted)
    ├── PIML_training_history.png    # Loss curves (MSE + physics components)
    ├── PIML_uncertainty.png         # MC-Dropout uncertainty bands
    ├── PIML_shap.png                # SHAP feature importance
    ├── PIML_optimal_composition.png # Zn/Co composition heatmap
    ├── PIML_redox_peaks.png         # Peak current analysis
    ├── PIML_extrapolation.png       # Extrapolation analysis (4 panels)
    ├── PIML_butler_volmer_analysis.png  # Butler-Volmer & charge conservation
    ├── PIML_physics_dashboard.png       # 9-panel publication figure
    ├── PIML_physics_report.txt          # Text-based physics validation report
    ├── model_comparison_scatter.png # All models scatter comparison
    ├── model_comparison_metrics.png # R², RMSE, MAE bar charts
    ├── model_comparison_violations.png # Constraint violation comparison
    ├── model_comparison_table.csv   # Comparison table (CSV)
    ├── model_comparison_table.tex   # Comparison table (LaTeX)
    └── piml_full_results.json       # Complete evaluation results
```

---

## Physics Constraints (9 Total)

| # | Constraint | Equation | Implementation | \(\lambda\) |
|---|---|---|---|---|
| 1 | **Faraday's Law** | $Q = n \cdot F \cdot \Delta C$ | Penalises charge-concentration inconsistency | 0.001 |
| 2 | **Capacitance** | $C_{sp} = \int|I|dV / (2 \cdot m \cdot v \cdot \Delta V)$ | Enforces non-negative capacitance, variance penalty on $|I|/v$ | 0.001 |
| 3 | **Redox Limits** | $|I| \leq I_{max}$ | Penalises physically impossible currents | 0.001 |
| 4 | **Smoothness** | L2 regularisation on output | Prevents non-physical discontinuities | 0.001 |
| 5 | **Butler-Volmer** | $I = I_0[e^{\alpha nF\eta/RT} - e^{-(1-\alpha)nF\eta/RT}]$ | Enforces exponential current-overpotential relationship near $E^0$ | 0.0005 |
| 6 | **Nernst Equation** | $E = E^0 + (RT/nF)\ln(C_{ox}/C_{red})$ | Temperature dependence: $|I|/\sqrt{T}$ should be T-invariant | 0.0005 |
| 7 | **Charge Conservation** | $\oint I \cdot dV \approx 0$ | Anodic charge should balance cathodic charge | 0.001 |
| 8 | **Randles-Sevcik** | $I_p = 0.4463 n^{3/2} F A \sqrt{DFv/(RT)} C$ | $|I|/\sqrt{v}$ should be constant | 0.0005 |
| 9 | **Thermodynamic** | $\Delta G = -nFE$ | Current sign must match overpotential sign for $|\eta| > 50$ mV | 0.0002 |

### Curriculum Learning Strategy

Training uses a **warmup schedule** to ensure stable convergence:

| Phase | Epochs | Physics Weight | Description |
|---|---|---|---|
| Warmup | 1 → 80 | 0.0 | Pure MSE — learn data distribution first |
| Ramp | 80 → 160 | 0.0 → 1.0 | Linearly introduce physics constraints |
| Full | 160+ | 1.0 | Full hybrid loss (MSE + all physics terms) |

---

## Model Architectures

| Model | Type | Physics-Informed | Uncertainty | Data Augmentation |
|---|---|---|---|---|
| **PIML ANN** | Dense + MC Dropout | ✅ Full physics loss | ✅ MC Dropout | ✅ |
| **PIML LSTM** | Sequence model | ✅ Full physics loss | ❌ | ✅ |
| **PIML Transformer** | Attention-based | ✅ Full physics loss | ❌ | ✅ |
| Baseline ANN | Dense network | ❌ MSE only | ❌ | ❌ |
| Random Forest | Ensemble | ❌ | ❌ | ❌ |
| XGBoost | Gradient boosting | ❌ | ❌ | ❌ |
| Meta-Model | Stacking ensemble | ❌ | ❌ | ❌ |

---

## Data Augmentation Strategies

The `augment_dataset()` function applies physics-aware augmentation:

1. **Gaussian Noise Injection** — Small noise on current density and features (preserves distribution shape)
2. **Randles-Sevcik Scan-Rate Scaling** — Generate samples at new scan rates using $I_{new} = I_{old} \sqrt{v_{new}/v_{old}}$
3. **Composition Interpolation** — SMOTE-like interpolation between existing Zn/Co compositions to fill gaps

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook PIML_notebook.ipynb
```

Run all cells sequentially for the full pipeline: data generation → augmentation →
model training → evaluation → extrapolation analysis → comparison → publication output.

### Option 2: Command Line Training

```bash
cd main_files

# Train with synthetic data (default)
python PIML_training.py

# Train with experimental data
python PIML_training.py --use-experimental

# Train with custom parameters (including new physics constraints)
python PIML_training.py --arch ann --epochs 500 --lr 0.005 --lambda-faraday 0.001 --lambda-bv 0.0005 --lambda-nernst 0.0005 --lambda-charge 0.001 --lambda-randles 0.0005
```

### Option 3: Interactive Menu

```bash
cd main_files
python unified_model_interface.py
```

---

## Input Features

| Feature | Description | Unit |
|---|---|---|
| Potential | Applied voltage | V |
| OXIDATION | Oxidation state flag | 0/1 |
| Zn_Co_Conc | Total dopant concentration | mol fraction |
| Scan_Rate | CV scan rate | mV/s |
| ZN | Zinc dopant flag | 0/1 |
| CO | Cobalt dopant flag | 0/1 |
| Temperature | Electrode temperature | K |
| Electrode_Area | Working electrode area | cm² |

## Target Variables

| Target | Description | Unit |
|---|---|---|
| Current_Density | CV current density | mA/cm² |
| Specific_Capacitance | Gravimetric capacitance | F/g |
| Redox_Peak_Ox / Red | Redox peak potentials | V |

---

## Evaluation Metrics

- **R²** — Coefficient of determination
- **RMSE** — Root mean squared error
- **MAE** — Mean absolute error
- **Physics Constraint Violations** — 7 violation metrics across 9 constraints:
  - Current exceed % (redox limits)
  - Negative capacitance %
  - Large jumps % (smoothness)
  - Butler-Volmer sign violation %
  - Charge imbalance %
  - Nernst temperature violation %
  - Randles-Sevcik R²
- **95% CI coverage** — Calibration of MC-Dropout uncertainty estimates
- **Tafel slope analysis** — Anodic and cathodic Tafel slopes from predicted I–η curves
- **SHAP values** — Feature attribution for interpretability
- **Physics Radar Score** — Overall constraint satisfaction visualised as radar chart

---

## Extrapolation Analysis

A key advantage of physics-informed models is physically plausible extrapolation.
The evaluation includes:

1. **Scan Rate Extrapolation** — Predictions at 350–1000 mV/s (training max: 200 mV/s)
2. **Temperature Extrapolation** — Predictions at 348–398 K (training max: 328 K)
3. **Novel Compositions** — Unseen Zn/Co ratios (e.g., Zn=Co=0.25)
4. **Randles-Sevcik Consistency** — Verification that peak current scales as $\sqrt{v}$ even in extrapolation

---

## Reproducibility

All code is deterministic with fixed random seeds (`seed=42`). To reproduce results:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook PIML_notebook.ipynb`
4. Execute all cells — models will train from scratch and regenerate all plots

---

## Requirements

- Python >= 3.10
- TensorFlow >= 2.16
- See `requirements.txt` for full list

---

## Citation

If you use this code, please cite the original experimental work:

```bibtex
@article{bifeO3_cv_ml,
    title={Machine Learning-Based Prediction of Cyclic Voltammetry Behavior of
           Substitution of Zinc and Cobalt in BiFeO3/Bi25FeO40 for
           Supercapacitor Applications},
    journal={ACS Omega},
    year={2024},
    doi={10.1021/acsomega.3c10485}
}
```

---

## License

This project is provided for academic and research purposes.


```bash


========== FINAL ML METRICS ==========
R²         = 0.9854
RMSE       = 0.0432
MAE        = 0.0211
MSE        = 0.0018
=======================================
``` 
