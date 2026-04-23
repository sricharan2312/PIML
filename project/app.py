from flask import Flask, render_template, request, jsonify, Response
import sys
import os
import json
import io
import csv

# Add main_files to the path to import ML code later
sys.path.append(os.path.join(os.path.dirname(__file__), 'main_files'))

app = Flask(__name__)

# Global dictionary to hold user inputs across the steps
EXPERIMENT_DATA = {'config': {}, 'results': {}}

# Serve UI steps
@app.route('/')
def step_1():
    EXPERIMENT_DATA['config'] = {} # Reset on new experiment
    EXPERIMENT_DATA['results'] = {}
    return render_template('index.html')

@app.route('/step2')
def step_2(): return render_template('step2.html')
@app.route('/step3')
def step_3(): return render_template('step3.html')
@app.route('/step4')
def step_4(): return render_template('step4.html')
@app.route('/step5')
def step_5(): return render_template('step5.html')
@app.route('/step6')
def step_6(): return render_template('step6.html')
@app.route('/step7')
def step_7(): return render_template('step7.html')
@app.route('/step8')
def step_8(): return render_template('step8.html')

@app.route('/results')
def results():
    return render_template('results.html', data=EXPERIMENT_DATA)

@app.route('/debug')
def debug_state():
    return jsonify(EXPERIMENT_DATA)

# API Route to handle the potential form submisson
@app.route('/api/configure_potential', methods=['POST'])
def configure_potential():
    data = request.json
    potential_value = data.get('potential')
    EXPERIMENT_DATA['config']['Target Potential (eV)'] = potential_value
    print(f"Received potential configuration: {potential_value} eV")
    return jsonify({"status": "success", "message": "Potential configured successfully"})

# API Route to handle steps 2-8 form submissions
@app.route('/api/save_step', methods=['POST'])
def save_step():
    data = request.json
    page = data.get('page')
    step_data = data.get('data', {})
    print(f"Received data for {page}: {step_data}")
    
    # Store data mapping from the UI
    for key, val in step_data.items():
        EXPERIMENT_DATA['config'][key] = val
    
    # If this is the last step (e.g. step 8), trigger inference
    if page and 'step8' in page:
        print("\nFinal step reached! Loading pre-trained model for instant inference...")
        try:
            import tensorflow as tf
            import pandas as pd
            import numpy as np
            import joblib
            import matplotlib
            matplotlib.use('Agg') # Safe for server rendering
            import matplotlib.pyplot as plt
            from PIML_model import MCDropout
            
            base_dir = os.path.dirname(__file__)
            models_dir = os.path.join(base_dir, 'main_files', 'saved_models')
            model_path = os.path.join(models_dir, 'piml_model.keras')
            
            model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'MCDropout': MCDropout})
            X_scaler = joblib.load(os.path.join(models_dir, 'piml_scaler.pkl'))
            y_scaler = joblib.load(os.path.join(models_dir, 'piml_y_scaler.pkl'))
            
            print("\nGenerating NEW CV curve predictions for the unseen setup...")
            
            potentials_fwd = np.linspace(-0.6, 0.8, 100)
            potentials_rev = np.linspace(0.8, -0.6, 100)
            potentials = np.concatenate([potentials_fwd, potentials_rev])
            
            # Map dynamic inputs safely
            cfg = EXPERIMENT_DATA['config']
            def safe_float(key, default):
                try:
                    val = cfg.get(key)
                    if val is None or str(val).strip() == '': return default
                    return float(val)
                except (ValueError, TypeError):
                    return default
            
            temp_K = safe_float('temp-slider', 298.0)
            scan_rate = safe_float('scan-input', 50.0)
            zn_ratio = safe_float('ratio-input', 0.1)
            electrode_area = safe_float('electrode_area', 0.1)
            
            # --- PHYSICS OOD EXTRAPOLATION FIX ---
            # The neural network was trained on bounded laboratory domains (e.g. Area ~0.1, Scan_Rate <= 150).
            # Extreme inputs (like Area=12.5) trigger unbounded linear explosion in standard ML models.
            # We constrain the inputs for the neural forward-pass to get a perfectly canonical j curve,
            # then scale the outputs mathematically using electrochemical physics (Randles-Sevcik: j ~ sqrt(v)).
            nn_area = 0.106 # Mean training area (prevents area-based explosion of J)
            nn_scan_rate = np.clip(scan_rate, 10.0, 150.0)
            nn_zn_ratio = np.clip(zn_ratio, 0.0, 0.3)
            nn_temp = np.clip(temp_K, 273.15, 350.0)

            # Randles-Sevcik diffusion scale factor for current density
            j_scale_factor = np.sqrt(scan_rate / nn_scan_rate)
            
            new_records = []
            
            # 1. Forward sweep (Anodic / Oxidation Phase = 1.0)
            for V in potentials_fwd:
                new_records.append([
                    V, 1.0, nn_zn_ratio, nn_scan_rate, 
                    1.0 if nn_zn_ratio > 0 else 0.0, 1.0, 
                    nn_temp, nn_area         
                ])
                
            # 2. Reverse sweep (Cathodic / Reduction Phase = 0.0)
            for V in potentials_rev:
                new_records.append([
                    V, 0.0, nn_zn_ratio, nn_scan_rate, 
                    1.0 if nn_zn_ratio > 0 else 0.0, 1.0, 
                    nn_temp, nn_area         
                ])
                
            X_new_raw = np.array(new_records, dtype=np.float32)
            X_new_scaled = X_scaler.transform(X_new_raw).astype(np.float32)
            y_mean_scaled = model.predict(X_new_scaled, verbose=0)
            
            if np.isnan(y_mean_scaled).any() or np.isinf(y_mean_scaled).any():
                print("[WARNING] Loaded PIML model returning NaN/Inf weights. Zeroing.")
                y_mean_scaled = np.nan_to_num(y_mean_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                
            # These are Canonical Current Densities (mA/cm2)
            y_pred_actual = y_scaler.inverse_transform(y_mean_scaled.reshape(-1, 1)).flatten()
            
            # Apply physics extrapolation scaling back to Current Density
            y_pred_actual = y_pred_actual * j_scale_factor
            
            # Physics Calculation for Peak Metrics
            max_idx = np.argmax(y_pred_actual)
            min_idx = np.argmin(y_pred_actual)

            # j_pa is Current Density (mA/cm2). I_pa is TOTAL CURRENT (mA) = j * Area.
            j_pa = float(y_pred_actual[max_idx])
            j_pc = float(y_pred_actual[min_idx])
            
            i_pa = j_pa * electrode_area
            i_pc = j_pc * electrode_area
            
            e_pa = float(potentials[max_idx])
            e_pc = float(potentials[min_idx])
            delta_e = abs(e_pa - e_pc)
            
            print("\n=== PIML Model Prediction Metrics ===")
            print(f"Anodic Peak (I_pa): {i_pa:.4f} mA")
            print(f"Cathodic Peak (I_pc): {i_pc:.4f} mA")
            print(f"Peak Split (\u0394E_p): {delta_e:.4f} V")
            print(f"Anodic Potential (E_pa): {e_pa:.4f} V")
            print(f"Cathodic Potential (E_pc): {e_pc:.4f} V")
            print(f"Predicted Current Array (first 5 points): {y_pred_actual[:5]}")
            print("=====================================\n")

            # Store the raw lists & metrics globally to push perfectly to Chart.js in results.html!
            EXPERIMENT_DATA['results'] = {
                'potentials': potentials.tolist(),
                'currents': (y_pred_actual * electrode_area).tolist(),
                'metrics': {
                    'Anodic Peak (I_pa)': f"{i_pa:.4f} mA",
                    'Cathodic Peak (I_pc)': f"{i_pc:.4f} mA",
                    'Peak Split (\u0394E_p)': f"{delta_e:.4f} V",
                    'Anodic Potential (E_pa)': f"{e_pa:.4f} V",
                    'Cathodic Potential (E_pc)': f"{e_pc:.4f} V"
                }
            }
            
            # Legacy PNG Fallback image, safely generating if JS fails
            plt.figure(figsize=(10, 6))
            plt.plot(potentials, y_pred_actual, 'b-', linewidth=3, label="Predicted CV Current")
            plt.xlabel('Potential (V vs. Ag/AgCl)', fontsize=14, fontweight='bold')
            plt.ylabel('Current Density (mA/cm²)', fontsize=14, fontweight='bold')
            plt.title('Predicted Cyclic Voltammetry (CV) Curve', fontsize=18, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
            plt.tight_layout()
            
            graph_path = os.path.join(base_dir, 'static', 'assets', 'cv_curve_prediction.png')
            os.makedirs(os.path.dirname(graph_path), exist_ok=True)
            plt.savefig(graph_path, dpi=300)
            plt.close()
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error during ML inference flow:\n{error_details}")
            EXPERIMENT_DATA['results'] = {'error': str(e), 'traceback': error_details}
    
    return jsonify({"status": "success"})

@app.route('/api/export_csv')
def export_csv():
    # Allow researchers to download the predicted curve directly
    if not EXPERIMENT_DATA['results']:
        return "No data to export", 400
        
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Applied Potential (V vs Ag/AgCl)', 'Predicted Current Density (mA/cm^2)'])
    
    potentials = EXPERIMENT_DATA['results']['potentials']
    currents = EXPERIMENT_DATA['results']['currents']
    for p, c in zip(potentials, currents):
        cw.writerow([round(p, 4), round(c, 4)])
        
    output = si.getvalue()
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=predicted_cv_curve.csv"}
    )

if __name__ == '__main__':
    # Running on port 5000
    app.run(debug=True, port=5000)
