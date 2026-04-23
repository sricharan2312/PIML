import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIML_model import build_piml_ann, PhysicsLoss, PIMLTrainer

df = pd.read_csv('Dataset/synthetic_cv_dataset.csv')
features = ['Potential', 'OXIDATION', 'Zn_Co_Conc', 'Scan_Rate', 'ZN', 'CO', 'Temperature', 'Electrode_Area']
X = df[features].values
y = df['Current_Density'].values
sx = StandardScaler(); X_s = sx.fit_transform(X)
sy = StandardScaler(); y_s = sy.fit_transform(y.reshape(-1,1)).flatten()

model = build_piml_ann(8)
physics = PhysicsLoss()
trainer = PIMLTrainer(model, physics, tf.keras.optimizers.Adam(5e-3), use_relobralo=True)

X_bs = tf.constant(X_s, dtype=tf.float32)
X_br = tf.constant(X, dtype=tf.float32)
y_bs = tf.constant(y_s, dtype=tf.float32)

ds = tf.data.Dataset.from_tensor_slices((X_bs, X_br, y_bs)).shuffle(10000).batch(256)

for step, (xb, xr, yb) in enumerate(ds):
    with tf.GradientTape() as tape:
        y_pred = model(xb, training=True)
        # Using pw=0.0 for warmup simulation
        loss, comps = physics.total_loss(yb, y_pred, xr, tf.constant(0.0), trainer.balancer.weights)
    
    grads = tape.gradient(loss, model.trainable_variables)
    if any(tf.reduce_any(tf.math.is_nan(g)) for g in grads if g is not None):
        print(f"Grads NaN at step {step}!")
        break
    
    trainer.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    losses = tf.stack([comps['faraday'], comps['capacitance'], comps['redox'], comps['smooth'], comps['butler_volmer'], comps['nernst'], comps['charge_conservation'], comps['randles_sevcik'], comps['thermodynamic']])
    trainer.balancer.update_weights(losses)

print("Finished debug loop.")
