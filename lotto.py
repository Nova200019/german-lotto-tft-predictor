import os
import numpy as np
import pandas as pd
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input,
    MultiHeadAttention, LayerNormalization, Add, Activation,
    Lambda, Embedding, Concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
import tensorflow.keras.backend as K

# Enable mixed precision for Apple M1/M2/M4 (Metal backend)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

np.random.seed(42)
tf.random.set_seed(42)

# Configure GPU/Apple Silicon memory growth
physical_devices = tf.config.list_physical_devices()
for dev in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(dev, True)
    except Exception:
        pass

MODEL_PATH = "best_lotto_model_all_vars_with_date.keras"

# Load full dataset (all variables)
url = "https://johannesfriedrich.github.io/LottoNumberArchive/Lottonumbers_tidy_complete.json"
data = pd.read_json(url, convert_dates=False)

# Group data by draw ID and variable
draw_dict = defaultdict(lambda: defaultdict(list))
draw_dates = {}

for _, row in data.iterrows():
    draw_id = row['id']
    variable = row['variable']
    value = row['value']
    draw_dict[draw_id][variable].append(value)
    draw_dates[draw_id] = row['date']

# Identify all variables present
all_variables = sorted({var for draws in draw_dict.values() for var in draws.keys()})

print(f"Variables found in dataset: {all_variables}")

# Filter draws with complete data: Lottozahl with exactly 6 numbers
usable_draws = []
for draw_id, vars_dict in draw_dict.items():
    if 'Lottozahl' in vars_dict and len(vars_dict['Lottozahl']) == 6:
        # Optional: check other vars if needed, here we accept missing others
        usable_draws.append(draw_id)

usable_draws = sorted(usable_draws)  # Sort for chronological order
print(f"Total usable draws with Lottozahl (6 numbers): {len(usable_draws)}")

# Parameters
timesteps = 20
numbers_per_draw = 6
vocab_size = 49  # Lotto numbers from 1 to 49
date_feat_dim = 6
embed_dim = 16

# Function to encode date features as cyclic continuous values
def encode_date_features(dates):
    dates = pd.to_datetime(dates, format='%d.%m.%Y')
    day_of_week = dates.dayofweek.values
    day_of_month = dates.day.values
    month = dates.month.values

    day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
    day_of_month_sin = np.sin(2 * np.pi * day_of_month / 31)
    day_of_month_cos = np.cos(2 * np.pi * day_of_month / 31)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    return np.stack([day_of_week_sin, day_of_week_cos,
                     day_of_month_sin, day_of_month_cos,
                     month_sin, month_cos], axis=1)

# Prepare sequences: inputs for all variables + dates, targets = next draw Lottozahl
X_vars_seq = {var: [] for var in all_variables}
dates_seq = []
y_main = []
used_draw_dates = []

# We create sequences of length timesteps, predicting next draw after that sequence
for i in range(len(usable_draws) - timesteps):
    seq_ids = usable_draws[i:i+timesteps]
    next_id = usable_draws[i+timesteps]

    # For each variable, prepare sequences of shape (timesteps, numbers_per_draw)
    for var in all_variables:
        seq_var = []
        for draw_id in seq_ids:
            numbers = draw_dict[draw_id].get(var, [])
            numbers = sorted(numbers)[:numbers_per_draw]
            # Pad with zeros if fewer than numbers_per_draw
            if len(numbers) < numbers_per_draw:
                numbers += [0] * (numbers_per_draw - len(numbers))
            # Shift by -1 for zero-based indexing, zeros remain zeros
            numbers = [(num - 1) if num > 0 else 0 for num in numbers]
            seq_var.append(numbers)
        X_vars_seq[var].append(seq_var)

    # Target: next draw's Lottozahl, sorted, zero-based
    next_numbers = sorted(draw_dict[next_id]['Lottozahl'])[:numbers_per_draw]
    next_numbers = [(num - 1) for num in next_numbers]
    y_main.append(next_numbers)

    # Dates sequence for timesteps
    seq_dates = [draw_dates[draw_id] for draw_id in seq_ids]
    dates_seq.append(encode_date_features(seq_dates))

    # Store date of the target (next draw) for reference
    used_draw_dates.append(draw_dates[next_id])

# Convert lists to numpy arrays
for var in all_variables:
    X_vars_seq[var] = np.array(X_vars_seq[var])  # shape (samples, timesteps, numbers_per_draw)
dates_seq = np.array(dates_seq)  # shape (samples, timesteps, date_feat_dim)
y_main = np.array(y_main)  # shape (samples, numbers_per_draw)

print(f"Total training samples prepared: {len(y_main)}")

# Multi-hot encode targets (for multi-label classification)
def to_multi_hot(numbers, size=vocab_size):
    one_hot = np.zeros(size, dtype=np.float32)
    for n in numbers:
        if 0 <= n < size:
            one_hot[n] = 1.0
    return one_hot

y_main_multi_hot = np.array([to_multi_hot(draw) for draw in y_main])

# Shuffle dataset
indices = np.arange(len(y_main_multi_hot))
np.random.shuffle(indices)
for var in all_variables:
    X_vars_seq[var] = X_vars_seq[var][indices]
dates_seq = dates_seq[indices]
y_main_multi_hot = y_main_multi_hot[indices]

print("\n📅 Example Lotto training samples with dates:")
for i in range(min(3, len(y_main_multi_hot))):
    numbers = np.where(y_main_multi_hot[i] == 1)[0] + 1
    print(f"{used_draw_dates[i]} -> {sorted(numbers.tolist())}")

# --- Model definition ---

def gated_residual_network(inputs, hidden_units, dropout_rate=0.1):
    x = Dense(hidden_units)(inputs)
    x = Activation('elu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(K.int_shape(inputs)[-1])(x)
    gating = Dense(K.int_shape(inputs)[-1], activation='sigmoid')(inputs)
    gated_x = x * gating
    return Add()([inputs, gated_x])

def temporal_fusion_transformer_block(inputs, num_heads=4, hidden_units=64, dropout_rate=0.1):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_units)(inputs, inputs)
    attn_output = Dropout(dropout_rate)(attn_output)
    attn_output = Add()([inputs, attn_output])
    attn_output = LayerNormalization()(attn_output)

    grn_output = gated_residual_network(attn_output, hidden_units, dropout_rate)
    grn_output = LayerNormalization()(grn_output)
    return grn_output

def build_tft_model_all_vars(
    timesteps, numbers_per_draw=6, vocab_size=49, embed_dim=16,
    date_feat_dim=6, variables=None
):
    assert variables is not None and len(variables) > 0, "Variables list cannot be empty"

    inputs_list = []
    embedded_vars = []

    # Create embeddings for each variable and average over numbers_per_draw
    for var in variables:
        var_input = Input(shape=(timesteps, numbers_per_draw), dtype='int32', name=f"{var}_input")
        inputs_list.append(var_input)
        x = Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)(var_input)
        # Average embeddings along the numbers_per_draw axis
        x = Lambda(lambda t: tf.reduce_mean(t, axis=2))(x)
        embedded_vars.append(x)  # shape (batch, timesteps, embed_dim)

    # Date input
    date_input = Input(shape=(timesteps, date_feat_dim), dtype='float32', name='date_input')
    inputs_list.append(date_input)

    # Concatenate embeddings and date features along last axis
    x = Concatenate(axis=-1)(embedded_vars + [date_input])

    # LSTM encoder
    x = LSTM(128, return_sequences=True)(x)

    # Stack 2 TFT blocks
    x = temporal_fusion_transformer_block(x, num_heads=4, hidden_units=128)
    x = temporal_fusion_transformer_block(x, num_heads=4, hidden_units=128)

    # Take output of last timestep
    x = Lambda(lambda t: t[:, -1, :])(x)

    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)

    # Output layer with sigmoid activation (multi-label)
    output = Dense(vocab_size, activation='sigmoid', name='main_output', dtype='float32')(x)

    model = Model(inputs=inputs_list, outputs=output)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

    return model

# Build or load the model
if os.path.exists(MODEL_PATH):
    print(f"\n📦 Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
else:
    model = build_tft_model_all_vars(
        timesteps, numbers_per_draw, vocab_size, embed_dim, date_feat_dim, all_variables
    )
    model.summary()

    checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss")
    early_stop = EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

    # Prepare input dictionary for training
    train_inputs = {f"{var}_input": X_vars_seq[var] for var in all_variables}
    train_inputs['date_input'] = dates_seq

    model.fit(
        train_inputs, y_main_multi_hot,
        epochs=200,
        batch_size=256,
        validation_split=0.1,
        callbacks=[early_stop, checkpoint, reduce_lr],
        verbose=1
    )

# --- Prediction with temperature scaling ---

def apply_temperature_scaling(probs, temperature=1.0):
    logits = np.log(probs + 1e-10) / temperature
    exp_logits = np.exp(logits)
    scaled_probs = exp_logits / np.sum(exp_logits)
    return scaled_probs

def predict_numbers(model, input_seq, temperature=1.0):
    pred_probs = model.predict(input_seq)[0]
    pred_probs = apply_temperature_scaling(pred_probs, temperature)

    # Choose top 6 numbers
    top_indices = np.argsort(pred_probs)[-6:][::-1]
    return sorted(top_indices + 1)

# Example: Predict next draw from the last 20 draws
last_seq_indices = usable_draws[-timesteps:]
input_for_pred = {}
for var in all_variables:
    seq_data = []
    for draw_id in last_seq_indices:
        nums = draw_dict[draw_id].get(var, [])
        nums = sorted(nums)[:numbers_per_draw]
        if len(nums) < numbers_per_draw:
            nums += [0] * (numbers_per_draw - len(nums))
        nums = [(n-1) if n > 0 else 0 for n in nums]
        seq_data.append(nums)
    input_for_pred[f"{var}_input"] = np.array([seq_data])

# Date features for prediction sequence
pred_dates = [draw_dates[d] for d in last_seq_indices]
input_for_pred['date_input'] = np.array([encode_date_features(pred_dates)])

predicted_nums = predict_numbers(model, input_for_pred, temperature=0.8)

print("\n Predicted next Lotto numbers:")
print(predicted_nums)
