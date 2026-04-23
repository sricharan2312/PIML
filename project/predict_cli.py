import argparse
import os
import sys

# Silence TensorFlow warnings for clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import joblib
import tensorflow as tf
from main_files.PIML_model import MCDropout

def main():
    parser = argparse.ArgumentParser(description='Predict CV current using PIML model')
    parser.add_argument('V', type=float, help='Target Potential (V vs Ag/AgCl)')
    parser.add_argument('oxidation_flag', type=float, help='Oxidation Flag (0 or 1)')
    parser.add_argument('zn_ratio', type=float, help='Zn/Co Ratio')
    parser.add_argument('scan_rate', type=float, help='Scan Rate (mV/s)')
    parser.add_argument('ZN_flag', type=float, help='Presence of Zn (0 or 1)')
    parser.add_argument('CO_flag', type=float, help='Presence of Co (0 or 1)')
    parser.add_argument('Temp', type=float, help='Temperature (K)')
    parser.add_argument('Area', type=float, help='Electrode Area (cm^2)')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'main_files', 'saved_models')
    model_path = os.path.join(models_dir, 'piml_model.keras')
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False, custom_objects={'MCDropout': MCDropout})
        X_scaler = joblib.load(os.path.join(models_dir, 'piml_scaler.pkl'))
        y_scaler = joblib.load(os.path.join(models_dir, 'piml_y_scaler.pkl'))
    except Exception as e:
        print(f"Error loading model or scalers: {e}")
        return

    # Extract features in exact expected order
    inputs = np.array([[
        args.V, args.oxidation_flag, args.zn_ratio, args.scan_rate,
        args.ZN_flag, args.CO_flag, args.Temp, args.Area
    ]], dtype=np.float32)
    
    try:
        X_scaled = X_scaler.transform(inputs).astype(np.float32)
        y_scaled = model.predict(X_scaled, verbose=0)
        
        if np.isnan(y_scaled).any() or np.isinf(y_scaled).any():
            y_scaled = np.nan_to_num(y_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            
        y_actual = y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()[0]
        
        print("\n=== PIML Model Prediction ===")
        print(f"Input Features:")
        print(f"  Potential:      {args.V} V")
        print(f"  Oxidation Flag: {args.oxidation_flag}")
        print(f"  Zn/Co Ratio:    {args.zn_ratio}")
        print(f"  Scan Rate:      {args.scan_rate} mV/s")
        print(f"  Zn Flag:        {args.ZN_flag}")
        print(f"  Co Flag:        {args.CO_flag}")
        print(f"  Temperature:    {args.Temp} K")
        print(f"  Electrode Area: {args.Area} cm^2")
        print("-" * 30)
        print(f"PREDICTED CURRENT: {y_actual:.4f} mA/cm^2")
        print("=============================\n")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
