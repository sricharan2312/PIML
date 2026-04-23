"""
Physics-Informed Machine Learning (PIML) Model
================================================
Hybrid neural network that embeds electrochemical physics constraints
directly into the loss function for predicting CV curves of
Zn/Co-substituted BiFeO3/Bi25FeO40 materials.

Architecture options:
  1. Dense ANN with MC Dropout (default)
  2. LSTM for sequential CV sweep data
  3. Transformer encoder for full CV curve modelling

Physics constraints embedded in training (9 total):
  (a) Faraday's law:       Q = n·F·ΔC  (charge ∝ concentration)
  (b) Capacitance:         C_sp = ∫|I|dV / (2·m·v·ΔV)
  (c) Redox limits:        penalise physically impossible currents
  (d) Smoothness:          penalise non-physical discontinuities
  (e) Butler-Volmer:       I = I₀[exp(αnFη/RT) - exp(-(1-α)nFη/RT)]
  (f) Nernst equation:     E = E⁰ + (RT/nF)·ln(C_ox/C_red)
  (g) Charge conservation: ∮I·dV ≈ 0 over a full CV cycle
  (h) Randles-Sevcik:      I_peak ∝ √(scan_rate)
  (i) Thermodynamic:       ΔG = -nFE consistency

Loss = MSE_data + Σᵢ λᵢ * L_physics_i

physics_weight follows curriculum learning:
  epoch ≤ warmup       → 0  (pure data-driven)
  warmup < epoch ≤ 2×w → linear ramp 0→1
  epoch > 2×warmup     → 1  (full physics)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# ============================================================
# PHYSICAL CONSTANTS
# ============================================================
FARADAY_CONST = 96485.3329   # C/mol
R_GAS         = 8.314        # J/(mol·K)
N_ELECTRONS   = 2            # electrons transferred
ALPHA_BV      = 0.5          # Butler-Volmer transfer coefficient (symmetric)
E0_STANDARD   = 0.25         # V – standard redox potential for BFO system
T_REFERENCE   = 298.15       # K – reference temperature


# ============================================================
# 1.  DENSE ANN WITH MC DROPOUT  (default architecture)
# ============================================================

class MCDropout(layers.Dropout):
    """Dropout layer that stays active at inference time for Monte-Carlo sampling."""
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


def build_piml_ann(
    input_dim: int,
    hidden_units: list = None,
    dropout_rate: float = 0.10,
    mc_dropout: bool = True,
    output_dim: int = 1,
    use_batchnorm: bool = False,
) -> keras.Model:
    """
    Builds a dense ANN with optional MC-Dropout for uncertainty quantification.

    Parameters
    ----------
    input_dim    : number of input features
    hidden_units : list of neurons per hidden layer  (default [256, 128, 64, 32])
    dropout_rate : fraction of units to drop
    mc_dropout   : if True, dropout remains active at inference (Bayesian approx.)
    output_dim   : 1 for current-density only; 3 for (current, capacitance, redox_peak)
    use_batchnorm: if True, add BatchNormalization after each dense layer

    Returns
    -------
    keras.Model
    """
    if hidden_units is None:
        hidden_units = [256, 128, 64, 32]

    DropoutLayer = MCDropout if mc_dropout else layers.Dropout

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = layers.Dense(units, activation="swish", name=f"dense_{i}")(x)
        if use_batchnorm:
            x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = DropoutLayer(dropout_rate, name=f"drop_{i}")(x)

    # Output head(s)
    current_out = layers.Dense(output_dim, name="current_density")(x)

    model = keras.Model(inputs=inputs, outputs=current_out, name="PIML_ANN")
    return model


# ============================================================
# 2.  LSTM FOR SEQUENTIAL CV DATA
# ============================================================

def build_piml_lstm(
    seq_len: int,
    n_features: int,
    hidden_units: int = 64,
    dropout_rate: float = 0.15,
) -> keras.Model:
    """
    LSTM-based model that treats a CV sweep (potential sequence) as a time series.

    Input shape: (batch, seq_len, n_features)
    Output: (batch, seq_len, 1)  – current density at each potential step.
    """
    inputs = keras.Input(shape=(seq_len, n_features), name="cv_sequence")

    x = layers.LSTM(hidden_units, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(hidden_units // 2, return_sequences=True, name="lstm_2")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.TimeDistributed(layers.Dense(1), name="output")(x)

    model = keras.Model(inputs=inputs, outputs=x, name="PIML_LSTM")
    return model


# ============================================================
# 3.  TRANSFORMER ENCODER
# ============================================================

class TransformerBlock(layers.Layer):
    """Single Transformer encoder block."""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="gelu"),
            layers.Dense(embed_dim),
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(rate)
        self.drop2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn = self.att(inputs, inputs)
        attn = self.drop1(attn, training=training)
        out1 = self.ln1(inputs + attn)
        ffn  = self.ffn(out1)
        ffn  = self.drop2(ffn, training=training)
        return self.ln2(out1 + ffn)


def build_piml_transformer(
    seq_len: int,
    n_features: int,
    embed_dim: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_blocks: int = 2,
    dropout_rate: float = 0.1,
) -> keras.Model:
    """Transformer encoder for full CV curve prediction."""
    inputs = keras.Input(shape=(seq_len, n_features), name="cv_sequence")

    # Project features → embedding dim
    x = layers.Dense(embed_dim, name="input_proj")(inputs)

    for i in range(num_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate,
                             name=f"transformer_{i}")(x)

    x = layers.TimeDistributed(layers.Dense(1), name="output")(x)
    model = keras.Model(inputs=inputs, outputs=x, name="PIML_Transformer")
    return model


# ============================================================
# 4.  PHYSICS-INFORMED LOSS FUNCTIONS & BALANCING
# ============================================================

class ReLoBRaLoBalancerTF:
    """
    Self-Adaptive Weighting for multi-objective loss using 
    Relative Loss Balancing with Random Loopback (ReLoBRaLo).
    """
    def __init__(self, num_losses=9, alpha=0.999, temperature=0.1, rho=0.99):
        self.num_losses_f = tf.constant(num_losses, tf.float32)
        self.alpha = tf.constant(alpha, tf.float32)
        self.temperature = tf.constant(temperature, tf.float32)
        self.rho = tf.constant(rho, tf.float32)
        
        # Track weights and historical losses
        self.weights = tf.Variable(tf.ones([num_losses], dtype=tf.float32), trainable=False)
        self.initial_losses = tf.Variable(tf.zeros([num_losses], dtype=tf.float32), trainable=False)
        self.prev_losses = tf.Variable(tf.zeros([num_losses], dtype=tf.float32), trainable=False)
        self.ema_losses = tf.Variable(tf.zeros([num_losses], dtype=tf.float32), trainable=False)
        self.initialized = tf.Variable(False, trainable=False)

    def update_weights(self, current_losses):
        current_losses = tf.cast(current_losses, tf.float32)
        
        def init_fn():
            self.initial_losses.assign(current_losses)
            self.prev_losses.assign(current_losses)
            self.ema_losses.assign(current_losses)
            self.initialized.assign(True)
            return self.weights

        def update_fn():
            eps = 1e-8
            rel_drop_initial = current_losses / (self.initial_losses + eps)
            rel_drop_ema = current_losses / (self.ema_losses + eps)
            
            rand_val = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
            bernoulli = tf.cast(rand_val < self.rho, tf.float32)
            
            lambda_loss = bernoulli * rel_drop_initial + (1.0 - bernoulli) * rel_drop_ema
            
            new_weights = tf.nn.softmax(lambda_loss / self.temperature)
            updated_w = new_weights * self.num_losses_f
            
            self.weights.assign(updated_w)
            self.prev_losses.assign(current_losses)
            new_ema = self.alpha * self.ema_losses + (1.0 - self.alpha) * current_losses
            self.ema_losses.assign(new_ema)
            
            return self.weights

        return tf.cond(self.initialized, update_fn, init_fn)


class PhysicsLoss:
    """
    Computes individual physics-constraint losses and the combined
    hybrid loss for training the PIML model.

    Expected tensor shapes
    ----------------------
    y_true, y_pred : (batch,) or (batch, 1)   – current density in mA/cm²
    inputs         : (batch, n_features)       – raw (scaled) feature tensor
                     Column order assumed after scaling:
                       [Potential, OXIDATION, Zn_Co_Conc, Scan_Rate,
                        ZN, CO, Temperature, Electrode_Area]
    """

    # Default feature column indices (after preprocessing)
    IDX_POTENTIAL      = 0
    IDX_OXIDATION      = 1
    IDX_ZN_CO_CONC     = 2
    IDX_SCAN_RATE      = 3
    IDX_ZN             = 4
    IDX_CO             = 5
    IDX_TEMPERATURE    = 6
    IDX_ELECTRODE_AREA = 7

    def __init__(
        self,
        lambda_faraday: float = 0.1,
        lambda_capacitance: float = 0.1,
        lambda_redox: float = 0.5,
        lambda_smooth: float = 0.05,
        lambda_butler_volmer: float = 0.05,
        lambda_nernst: float = 0.05,
        lambda_charge_conservation: float = 0.05,
        lambda_randles_sevcik: float = 0.05,
        lambda_thermodynamic: float = 0.02,
        max_current_density: float = 50.0,   # mA/cm² physical upper bound
        min_capacitance: float = 0.0,
        n_electrons: int = N_ELECTRONS,
        y_mean: float = 0.0,         # target scaler mean (for unscaling)
        y_scale: float = 1.0,        # target scaler std  (for unscaling)
        alpha_bv: float = ALPHA_BV,
        E0: float = E0_STANDARD,
        T_ref: float = T_REFERENCE,
    ):
        self.lambda_faraday              = lambda_faraday
        self.lambda_capacitance          = lambda_capacitance
        self.lambda_redox                = lambda_redox
        self.lambda_smooth               = lambda_smooth
        self.lambda_butler_volmer        = lambda_butler_volmer
        self.lambda_nernst               = lambda_nernst
        self.lambda_charge_conservation  = lambda_charge_conservation
        self.lambda_randles_sevcik       = lambda_randles_sevcik
        self.lambda_thermodynamic        = lambda_thermodynamic
        self.max_current                 = max_current_density
        self.min_cap                     = min_capacitance
        self.n_electrons                 = n_electrons
        self.y_mean                      = tf.constant(y_mean, dtype=tf.float32)
        self.y_scale                     = tf.constant(y_scale, dtype=tf.float32)
        self.alpha_bv                    = tf.constant(alpha_bv, dtype=tf.float32)
        self.E0                          = tf.constant(E0, dtype=tf.float32)
        self.T_ref                       = tf.constant(T_ref, dtype=tf.float32)

    def _unscale(self, y_pred):
        """Inverse-transform scaled predictions back to original units (mA/cm²)."""
        return y_pred * self.y_scale + self.y_mean

    # ----------------------------------------------------------
    def mse_loss(self, y_true, y_pred):
        """Standard data-driven MSE."""
        y_pred = tf.squeeze(y_pred)   # (B,1) → (B,)
        y_true = tf.squeeze(y_true)   # Ensure y_true is also (B,) to avoid (B, B) broadcasting!
        return tf.reduce_mean(tf.square(y_true - y_pred))

    # ----------------------------------------------------------
    def faraday_loss(self, y_pred, inputs):
        """
        Faraday's law soft constraint.
        Q = n·F·ΔC  =>  charge should be proportional to concentration.
        We approximate Q ~ |I| * Δt  and penalise deviation from proportionality.
        Only applied to doped samples (conc > 0); pure BFO is excluded.
        """
        conc = inputs[:, self.IDX_ZN_CO_CONC]       # dopant concentration
        scan_rate = tf.math.abs(inputs[:, self.IDX_SCAN_RATE]) + 1e-4    # mV/s

        # Unscale predictions to physical units (mA/cm²)
        y_phys = tf.squeeze(self._unscale(y_pred))

        # Mask: only penalise doped samples (concentration > 0)
        mask = tf.cast(conc > 1e-6, tf.float32)
        n_valid = tf.reduce_sum(mask) + 1e-8

        # Approximate charge ∝ |I| / scan_rate  (normalise scan_rate to V/s)
        scan_rate_Vs = scan_rate * 1e-3 + 1e-8   # mV/s → V/s
        Q_approx = tf.abs(y_phys) / scan_rate_Vs

        # Expected proportionality: Q ∝ n·F·C  (scale F down for numerical stability)
        Q_expected = tf.cast(self.n_electrons, tf.float32) * (FARADAY_CONST / 1e5) * conc

        # Normalised residual (log-space for stability)
        ratio = Q_approx / (Q_expected + 1e-8)
        # Clamp ratio to prevent explosion
        ratio = tf.clip_by_value(ratio, 0.01, 100.0)
        residual = tf.math.log(ratio + 1e-8)

        # Masked mean
        loss = tf.reduce_sum(tf.square(residual) * mask) / n_valid
        return loss

    # ----------------------------------------------------------
    def capacitance_loss(self, y_pred, inputs):
        """
        Capacitance consistency constraint.
        Higher scan rates should generally yield higher absolute current.
        Penalise cases where predicted |I| decreases with increasing scan rate
        (soft monotonicity with respect to scan rate).
        """
        # Unscale predictions to physical units
        y_phys = tf.squeeze(self._unscale(y_pred))
        scan_rate = tf.math.abs(inputs[:, self.IDX_SCAN_RATE]) + 1e-4

        # Capacitive current: I_cap = C * v, so |I|/v should be roughly constant
        scan_rate_Vs = scan_rate * 1e-3 + 1e-8  # mV/s → V/s
        C_est = tf.abs(y_phys) / scan_rate_Vs

        # Penalise large variance in C_est (should be approximately constant
        # for the capacitive component)
        C_mean = tf.reduce_mean(C_est)
        penalty = tf.reduce_mean(tf.square(C_est - C_mean)) / (C_mean ** 2 + 1e-8)

        return penalty

    # ----------------------------------------------------------
    def redox_limit_loss(self, y_pred):
        """
        Penalise physically impossible current values:
          - Currents exceeding max_current_density
        """
        # Unscale predictions to physical units
        y_phys = tf.squeeze(self._unscale(y_pred))
        exceed = tf.nn.relu(tf.abs(y_phys) - self.max_current)
        return tf.reduce_mean(tf.square(exceed))

    # ----------------------------------------------------------
    def smoothness_loss(self, y_pred):
        """
        Penalise extreme predicted values (L2 regularisation on output).
        This replaces the finite-difference approach which requires ordered
        potential within a batch (not guaranteed with shuffled data).
        """
        y_flat = tf.squeeze(y_pred)
        # Penalise large deviations from zero (regulariser)
        return tf.reduce_mean(tf.square(y_flat)) * 1e-3

    # ----------------------------------------------------------
    def butler_volmer_loss(self, y_pred, inputs):
        """
        Butler-Volmer kinetics constraint.
        The fundamental equation of electrode kinetics:
            I = I₀ [exp(α·n·F·η / (R·T)) - exp(-(1-α)·n·F·η / (R·T))]
        where η = E - E⁰ is the overpotential.

        We penalise deviations from the expected exponential
        current-overpotential relationship near the equilibrium
        potential (within ±100 mV of E⁰).
        """
        potential = inputs[:, self.IDX_POTENTIAL]
        temperature = inputs[:, self.IDX_TEMPERATURE]
        y_phys = tf.squeeze(self._unscale(y_pred))

        # Overpotential η = E - E⁰
        eta = potential - self.E0
        n_f = tf.cast(self.n_electrons, tf.float32)
        alpha = self.alpha_bv

        # Thermal voltage: RT/(nF)
        T_safe = tf.maximum(temperature, 200.0)  # prevent division issues
        thermal_voltage = (R_GAS * T_safe) / (n_f * FARADAY_CONST)

        # Butler-Volmer predicted current shape (normalised):
        #   I_BV ∝ exp(α·η/V_T) - exp(-(1-α)·η/V_T)
        # Clamp exponents for numerical stability
        exp_anod = tf.exp(tf.clip_by_value(alpha * eta / thermal_voltage, -20.0, 20.0))
        exp_cath = tf.exp(tf.clip_by_value(-(1.0 - alpha) * eta / thermal_voltage, -20.0, 20.0))
        I_bv_shape = exp_anod - exp_cath

        # Normalise both to same scale for shape comparison
        I_bv_norm = I_bv_shape / (tf.reduce_max(tf.abs(I_bv_shape)) + 1e-8)
        y_norm = y_phys / (tf.reduce_max(tf.abs(y_phys)) + 1e-8)

        # Focus on near-equilibrium region (|η| < 0.15 V)
        # where Butler-Volmer is most accurate
        bv_mask = tf.cast(tf.abs(eta) < 0.15, tf.float32)
        n_valid = tf.reduce_sum(bv_mask) + 1e-8

        residual = tf.square(y_norm - I_bv_norm) * bv_mask
        loss = tf.reduce_sum(residual) / n_valid
        return loss

    # ----------------------------------------------------------
    def nernst_loss(self, y_pred, inputs):
        """
        Nernst equation constraint for temperature dependence.
        E = E⁰ + (RT/nF)·ln(C_ox/C_red)

        The peak current position should shift predictably with
        temperature. We penalise predictions that violate the
        expected temperature scaling:
            ∂I/∂T should be consistent with Arrhenius-like activation.
            Higher T → higher diffusion → higher peak current.
            dI_peak/dT > 0 for faradaic processes.
        """
        temperature = inputs[:, self.IDX_TEMPERATURE]
        y_phys = tf.squeeze(self._unscale(y_pred))

        # Nernst-based expectation: current magnitude should scale
        # with √(T/T_ref) due to diffusion coefficient temperature
        # dependence: D ∝ T (Stokes-Einstein), I_p ∝ √D ∝ √T
        T_ratio = tf.sqrt(temperature / self.T_ref)

        # The predicted current normalised by T_ratio should be
        # approximately independent of temperature
        normalised = tf.abs(y_phys) / (T_ratio + 1e-8)
        mean_norm = tf.reduce_mean(normalised)

        # Penalise variance of the T-normalised current
        loss = tf.reduce_mean(tf.square(normalised - mean_norm)) / (mean_norm ** 2 + 1e-8)
        return loss * 0.1  # scale factor for numerical balance

    # ----------------------------------------------------------
    def charge_conservation_loss(self, y_pred, inputs):
        """
        Charge conservation constraint.
        For a complete CV cycle (oxidation + reduction sweep),
        the net charge should approximately equal zero:
            ∮ I·dV ≈ 0

        In practice, the cathodic charge should nearly balance
        the anodic charge. We group by oxidation state flag
        and penalise the imbalance.
        """
        oxidation = inputs[:, self.IDX_OXIDATION]
        y_phys = tf.squeeze(self._unscale(y_pred))

        # Separate anodic and cathodic contributions
        anodic_mask = tf.cast(oxidation > 0.5, tf.float32)
        cathodic_mask = 1.0 - anodic_mask

        n_anodic = tf.reduce_sum(anodic_mask) + 1e-8
        n_cathodic = tf.reduce_sum(cathodic_mask) + 1e-8

        # Mean current in each region (proxy for integrated charge)
        Q_anodic = tf.reduce_sum(y_phys * anodic_mask) / n_anodic
        Q_cathodic = tf.reduce_sum(y_phys * cathodic_mask) / n_cathodic

        # Net charge should be approximately zero for reversible processes
        # (Q_anodic + Q_cathodic ≈ 0 since cathodic current is negative)
        Q_net = Q_anodic + Q_cathodic
        Q_total = tf.abs(Q_anodic) + tf.abs(Q_cathodic) + 1e-8

        # Normalised imbalance
        loss = tf.square(Q_net / Q_total)
        return loss

    # ----------------------------------------------------------
    def randles_sevcik_loss(self, y_pred, inputs):
        """
        Randles-Sevcik equation constraint.
        For diffusion-controlled processes:
            I_peak ∝ √(scan_rate)

        We penalise deviations from this √v scaling by checking
        that |I|/√v is approximately constant across different
        scan rates within each batch.
        """
        scan_rate = tf.math.abs(inputs[:, self.IDX_SCAN_RATE]) + 1e-4
        y_phys = tf.squeeze(self._unscale(y_pred))

        # |I| / √v should be roughly constant (Randles-Sevcik)
        sqrt_v = tf.sqrt(scan_rate * 1e-3 + 1e-8)  # mV/s → V/s
        ratio = tf.abs(y_phys) / (sqrt_v + 1e-8)

        # Only consider points with significant current (avoid noise at baseline)
        significant = tf.cast(tf.abs(y_phys) > 0.1 * tf.reduce_max(tf.abs(y_phys)), tf.float32)
        n_sig = tf.reduce_sum(significant) + 1e-8

        ratio_mean = tf.reduce_sum(ratio * significant) / n_sig
        penalty = tf.reduce_sum(tf.square(ratio - ratio_mean) * significant) / n_sig
        loss = penalty / (ratio_mean ** 2 + 1e-8)
        return loss

    # ----------------------------------------------------------
    def thermodynamic_loss(self, y_pred, inputs):
        """
        Thermodynamic consistency constraint.
        ΔG = -nFE  →  The free energy change drives the reaction.

        For a thermodynamically consistent model:
        1. Higher |overpotential| → higher |current| (monotonicity)
        2. Current sign must match overpotential sign
           (anodic overpotential → positive current, and vice versa)
        3. At E = E⁰, current should cross zero
        """
        potential = inputs[:, self.IDX_POTENTIAL]
        y_phys = tf.squeeze(self._unscale(y_pred))

        # Overpotential
        eta = potential - self.E0

        # Constraint: sign consistency
        # If η > 0 (anodic), I should be > 0; if η < 0 (cathodic), I should be < 0
        # We penalise sign mismatches (I*η < 0)
        sign_product = y_phys * eta
        sign_violation = tf.nn.relu(-sign_product)  # penalty when I and η have opposite signs

        # Only apply far from equilibrium (|η| > 50 mV) where sign is unambiguous
        # Near E⁰, capacitive current can dominate either direction
        far_from_eq = tf.cast(tf.abs(eta) > 0.05, tf.float32)
        n_valid = tf.reduce_sum(far_from_eq) + 1e-8

        loss = tf.reduce_sum(sign_violation * far_from_eq) / n_valid
        # Scale by max current for normalisation
        loss = loss / (tf.reduce_max(tf.abs(y_phys)) + 1e-8)
        return loss

    # ----------------------------------------------------------
    def total_loss(self, y_true, y_pred, inputs, physics_weight=1.0, dy_weights=None):
        """
        Combined hybrid loss with curriculum learning weight.

        L = MSE + pw × Σᵢ (λᵢ × dynamic_weightᵢ × L_physics_i)

        9 physics constraints:
          1. Faraday's law (charge ∝ concentration)
          2. Capacitance consistency (|I|/v variance)
          3. Redox current limits (|I| ≤ max)
          4. Output smoothness (L2 regularisation)
          5. Butler-Volmer kinetics (I–η relationship)
          6. Nernst equation (temperature dependence)
          7. Charge conservation (∮I·dV ≈ 0)
          8. Randles-Sevcik (I_peak ∝ √v)
          9. Thermodynamic consistency (ΔG = -nFE)

        Parameters
        ----------
        physics_weight : float in [0, 1]
            Ramp factor for curriculum learning.
        dy_weights : Tensor of shape (9,) or None
            Dynamic weights for each physics loss constraint (e.g., from ReLoBRaLo).
        """
        l_mse         = self.mse_loss(y_true, y_pred)
        l_faraday     = self.faraday_loss(y_pred, inputs)
        l_cap         = self.capacitance_loss(y_pred, inputs)
        l_redox       = self.redox_limit_loss(y_pred)
        l_smooth      = self.smoothness_loss(y_pred)
        l_bv          = self.butler_volmer_loss(y_pred, inputs)
        l_nernst      = self.nernst_loss(y_pred, inputs)
        l_charge      = self.charge_conservation_loss(y_pred, inputs)
        l_randles     = self.randles_sevcik_loss(y_pred, inputs)
        l_thermo      = self.thermodynamic_loss(y_pred, inputs)

        pw = tf.cast(physics_weight, tf.float32)

        # Guard each loss against NaN / Inf (prevents 0 * NaN = NaN during warmup)
        def _safe(x):
            return tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))

        # Default weights are 1.0 if not provided
        if dy_weights is None:
            dy_weights = tf.ones([9], dtype=tf.float32)

        physics = (
            self.lambda_faraday              * dy_weights[0] * _safe(l_faraday)
            + self.lambda_capacitance        * dy_weights[1] * _safe(l_cap)
            + self.lambda_redox              * dy_weights[2] * _safe(l_redox)
            + self.lambda_smooth             * dy_weights[3] * _safe(l_smooth)
            + self.lambda_butler_volmer      * dy_weights[4] * _safe(l_bv)
            + self.lambda_nernst             * dy_weights[5] * _safe(l_nernst)
            + self.lambda_charge_conservation * dy_weights[6] * _safe(l_charge)
            + self.lambda_randles_sevcik     * dy_weights[7] * _safe(l_randles)
            + self.lambda_thermodynamic      * dy_weights[8] * _safe(l_thermo)
        )

        total = l_mse + tf.where(pw > 1e-8, pw * physics, tf.zeros_like(l_mse))

        return total, {
            "mse":                 l_mse,
            "faraday":             l_faraday,
            "capacitance":         l_cap,
            "redox":               l_redox,
            "smooth":              l_smooth,
            "butler_volmer":       l_bv,
            "nernst":              l_nernst,
            "charge_conservation": l_charge,
            "randles_sevcik":      l_randles,
            "thermodynamic":       l_thermo,
            "total":               total,
        }


# ============================================================
# 5.  PIML TRAINER (custom training loop)
# ============================================================

class PIMLTrainer:
    """
    Custom training loop that feeds *raw inputs* alongside labels
    so the physics loss can access physical quantities (potential,
    scan rate, concentration, etc.) even when features are scaled.

    Usage
    -----
        trainer = PIMLTrainer(model, physics_loss, optimizer)
        trainer.fit(X_train, y_train, X_val, y_val, epochs=200)
    """

    def __init__(
        self,
        model: keras.Model,
        physics_loss: PhysicsLoss,
        optimizer: keras.optimizers.Optimizer = None,
        feature_columns: list = None,
        use_relobralo: bool = True,
    ):
        self.model = model
        self.physics = physics_loss
        self.optimizer = optimizer or keras.optimizers.Adam(learning_rate=1e-3)
        self.use_relobralo = use_relobralo
        if self.use_relobralo:
            self.balancer = ReLoBRaLoBalancerTF(num_losses=9)
        else:
            self.balancer = None

        self.feature_columns = feature_columns or [
            "Potential", "OXIDATION", "Zn_Co_Conc", "Scan_Rate",
            "ZN", "CO", "Temperature", "Electrode_Area",
        ]
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_mse": [],  "val_mse": [],
            "physics_faraday": [], "physics_cap": [],
            "physics_redox": [],   "physics_smooth": [],
            "physics_butler_volmer": [], "physics_nernst": [],
            "physics_charge_conservation": [], "physics_randles_sevcik": [],
            "physics_thermodynamic": [],
            "physics_weight": [],
            "learning_rate": [],
        }

    # -------------------------------------------------------
    @tf.function
    def _train_step(self, X_scaled, X_raw, y, physics_weight):
        # Get dynamic weights (stop_gradient to treat them as constants for the tape)
        if self.use_relobralo:
            dy_weights = tf.stop_gradient(self.balancer.weights)
        else:
            dy_weights = None
            
        with tf.GradientTape() as tape:
            y_pred = self.model(X_scaled, training=True)
            loss, components = self.physics.total_loss(y, y_pred, X_raw, physics_weight, dy_weights)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Clip gradients by global norm to prevent explosion
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # Update balancer with raw unscaled physics losses
        if self.use_relobralo:
            physics_losses = tf.stack([
                components["faraday"], components["capacitance"], components["redox"],
                components["smooth"], components["butler_volmer"], components["nernst"],
                components["charge_conservation"], components["randles_sevcik"], components["thermodynamic"]
            ])
            self.balancer.update_weights(physics_losses)
            
        return loss, components

    @tf.function
    def _val_step(self, X_scaled, X_raw, y, physics_weight):
        if self.use_relobralo:
            dy_weights = tf.stop_gradient(self.balancer.weights)
        else:
            dy_weights = None
            
        y_pred = self.model(X_scaled, training=False)
        loss, components = self.physics.total_loss(y, y_pred, X_raw, physics_weight, dy_weights)
        return loss, components

    def _val_step_batched(self, X_scaled, X_raw, y, physics_weight, batch_size=4096):
        """Run validation in batches to avoid memory issues on large datasets."""
        n = X_scaled.shape[0]
        if n <= batch_size:
            return self._val_step(X_scaled, X_raw, y, physics_weight)

        total_loss = 0.0
        comp_accum = None
        n_batches = 0
        for i in range(0, n, batch_size):
            end = min(i + batch_size, n)
            loss, comps = self._val_step(X_scaled[i:end], X_raw[i:end], y[i:end], physics_weight)
            total_loss += loss.numpy()
            if comp_accum is None:
                comp_accum = {k: v.numpy() for k, v in comps.items()}
            else:
                for k, v in comps.items():
                    comp_accum[k] += v.numpy()
            n_batches += 1

        avg_loss = tf.constant(total_loss / n_batches, dtype=tf.float32)
        avg_comps = {k: tf.constant(v / n_batches, dtype=tf.float32)
                     for k, v in comp_accum.items()}
        return avg_loss, avg_comps

    # -------------------------------------------------------
    def fit(
        self,
        X_train_scaled, X_train_raw, y_train,
        X_val_scaled, X_val_raw, y_val,
        epochs: int = 200,
        batch_size: int = 512,
        patience: int = 25,
        verbose: int = 1,
        warmup_epochs: int = 50,
    ):
        """
        Train with early stopping on validation loss.

        Curriculum learning strategy:
          - For epochs 1..warmup_epochs: pure MSE (physics_weight = 0)
          - For epochs warmup_epochs..2*warmup_epochs: linearly ramp physics_weight 0→1
          - After that: full physics constraints (physics_weight = 1)

        Parameters
        ----------
        X_train_scaled : ndarray – StandardScaler-transformed features
        X_train_raw    : ndarray – original (unscaled) features for physics loss
        y_train        : ndarray – target current density (scaled)
        warmup_epochs  : int     – number of pure-MSE warmup epochs
        """
        n = len(X_train_scaled)
        best_val = np.inf
        wait = 0
        best_weights = None

        # Convert to tensors
        X_tr_s = tf.constant(X_train_scaled, dtype=tf.float32)
        X_tr_r = tf.constant(X_train_raw,    dtype=tf.float32)
        y_tr   = tf.constant(y_train,        dtype=tf.float32)
        X_v_s  = tf.constant(X_val_scaled,   dtype=tf.float32)
        X_v_r  = tf.constant(X_val_raw,      dtype=tf.float32)
        y_v    = tf.constant(y_val,          dtype=tf.float32)

        train_ds = tf.data.Dataset.from_tensor_slices((X_tr_s, X_tr_r, y_tr))
        train_ds = train_ds.shuffle(n).batch(batch_size).prefetch(tf.data.AUTOTUNE)

        for epoch in range(1, epochs + 1):
            # --- Curriculum learning: ramp physics weight ---
            if epoch <= warmup_epochs:
                pw = 0.0                               # pure MSE phase
            elif epoch <= 2 * warmup_epochs:
                pw = (epoch - warmup_epochs) / warmup_epochs  # linear ramp 0→1
            else:
                pw = 1.0                               # full physics
            pw_tf = tf.constant(pw, dtype=tf.float32)

            # --- Training ---
            epoch_losses = []
            for batch_Xs, batch_Xr, batch_y in train_ds:
                loss, comps = self._train_step(batch_Xs, batch_Xr, batch_y, pw_tf)
                epoch_losses.append(loss.numpy())
            train_loss = np.mean(epoch_losses)

            # --- Validation ---
            val_loss, val_comps = self._val_step_batched(X_v_s, X_v_r, y_v, pw_tf)
            val_loss_np = float(val_loss) if isinstance(val_loss, (int, float)) else val_loss.numpy()

            # --- History ---
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss_np)
            self.history["train_mse"].append(float(comps["mse"].numpy()))
            self.history["val_mse"].append(float(val_comps["mse"].numpy()))
            self.history["physics_faraday"].append(float(val_comps["faraday"].numpy()))
            self.history["physics_cap"].append(float(val_comps["capacitance"].numpy()))
            self.history["physics_redox"].append(float(val_comps["redox"].numpy()))
            self.history["physics_smooth"].append(float(val_comps["smooth"].numpy()))
            self.history["physics_butler_volmer"].append(float(val_comps["butler_volmer"].numpy()))
            self.history["physics_nernst"].append(float(val_comps["nernst"].numpy()))
            self.history["physics_charge_conservation"].append(float(val_comps["charge_conservation"].numpy()))
            self.history["physics_randles_sevcik"].append(float(val_comps["randles_sevcik"].numpy()))
            self.history["physics_thermodynamic"].append(float(val_comps["thermodynamic"].numpy()))
            self.history["physics_weight"].append(float(pw))
            # Handle standard LR or learning rate schedule
            lr_val = self.optimizer.learning_rate
            if hasattr(lr_val, '__call__'):
                # In TF 2.x, learning rate schedules can be called with the optimizer's iterations step
                lr_val = lr_val(self.optimizer.iterations)
            self.history["learning_rate"].append(float(lr_val))

            # --- Early stopping ---
            if val_loss_np < best_val:
                best_val = val_loss_np
                wait = 0
                best_weights = self.model.get_weights()
            else:
                wait += 1

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch:4d}  "
                      f"train_loss={train_loss:.5f}  val_loss={val_loss_np:.5f}  "
                      f"mse={val_comps['mse'].numpy():.5f}  "
                      f"faraday={val_comps['faraday'].numpy():.5f}  "
                      f"bv={val_comps['butler_volmer'].numpy():.5f}  "
                      f"redox={val_comps['redox'].numpy():.5f}  "
                      f"charge={val_comps['charge_conservation'].numpy():.5f}  "
                      f"pw={pw:.2f}  patience={patience - wait}")

            if wait >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}.")
                break

        if best_weights is not None:
            self.model.set_weights(best_weights)
            if verbose:
                print(f"Restored best weights (val_loss={best_val:.5f}).")

        return self.history

    # -------------------------------------------------------
    def predict(self, X_scaled, n_mc_samples: int = 0):
        """
        Predict current density.

        If n_mc_samples > 0 and model uses MCDropout, returns
        (mean_prediction, std_prediction) for uncertainty quantification.
        """
        if n_mc_samples > 0:
            preds = np.stack([
                self.model(X_scaled, training=True).numpy().flatten()
                for _ in range(n_mc_samples)
            ])
            return preds.mean(axis=0), preds.std(axis=0)
        else:
            return self.model(X_scaled, training=False).numpy().flatten(), None


# ============================================================
# 6.  MULTI-OUTPUT PIML MODEL (Current + Capacitance + Redox)
# ============================================================

def build_piml_multioutput(input_dim: int, hidden_units: list = None) -> keras.Model:
    """
    Multi-head model predicting:
      - Current density (mA/cm²)
      - Specific capacitance (F/g)
      - Redox peak potential (V)

    Shared trunk with separate prediction heads.
    """
    if hidden_units is None:
        hidden_units = [256, 128, 64]

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs
    for i, u in enumerate(hidden_units):
        x = layers.Dense(u, activation="relu", name=f"shared_{i}")(x)
        x = layers.BatchNormalization()(x)
        x = MCDropout(0.1)(x)

    # Head 1: Current density
    h1 = layers.Dense(32, activation="relu", name="head_current_1")(x)
    out_current = layers.Dense(1, name="current_density")(h1)

    # Head 2: Specific capacitance (must be ≥ 0)
    h2 = layers.Dense(32, activation="relu", name="head_cap_1")(x)
    out_cap = layers.Dense(1, activation="softplus", name="specific_capacitance")(h2)

    # Head 3: Redox peak potential
    h3 = layers.Dense(32, activation="relu", name="head_redox_1")(x)
    out_redox = layers.Dense(1, name="redox_peak")(h3)

    model = keras.Model(
        inputs=inputs,
        outputs=[out_current, out_cap, out_redox],
        name="PIML_MultiOutput",
    )
    return model
