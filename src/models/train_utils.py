import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tqdm.keras import TqdmCallback
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight

import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib 
import sys 

import warnings
warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BCE_LOSS = tf.keras.losses.BinaryCrossentropy()



# ---------------- Device ----------------
def get_device():
    gpus = tf.config.list_physical_devices("GPU")
    return "/GPU:0" if gpus else "/CPU:0"


# ---------------- Config ----------------
def load_config(config_path):
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file {config_path} not found")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

# ---------------- Data ---------------
def load_data(path):
    path = PROJECT_ROOT / path
    if not Path(path).exists():
        raise FileNotFoundError(f"File {path} not found")
    df = pd.read_csv(path)
    return df


def split_data(df, train_size=0.8):
    n_train = int(len(df) * train_size)
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    return train_df, test_df


def scale_data(train_df, test_df, scaler_path, target_col='label'):
    feature_cols = [c for c in train_df.columns if c != target_col]

    scaler = StandardScaler()

    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved in {scaler_path}")

    return train_df, test_df, scaler


def reshape_flat_windows(df, target_col="label"): # for the oversampling technique
    feature_cols = [c for c in df.columns if c != target_col and "_t" in c]

    X = df[feature_cols].astype(float).values
    y = df[target_col].astype(int).values

    timesteps = sorted({int(c.split("_t")[-1]) for c in feature_cols})
    n_steps = len(timesteps)

    base_features = sorted({c.rsplit("_t", 1)[0] for c in feature_cols})
    n_features = len(base_features)

    if X.shape[1] != n_steps * n_features:
        raise ValueError(f"Expected {n_steps * n_features} feature columns, " f"but got {X.shape[1]}")

    X = X.reshape(X.shape[0], n_steps, n_features)
    return X, y


def create_time_windows(df, n_steps=4, target_col='label'):
    feature_cols = [c for c in df.columns if c != target_col]

    X_values = df[feature_cols].to_numpy(dtype=np.float32)
    y_values = df[target_col].to_numpy(dtype=np.int64)

    X, y = [], []
    for i in range(len(X_values) - n_steps):
        X.append(X_values[i:i+n_steps])
        y.append(y_values[i + n_steps])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)

def split_windows_train_val(X, y, val_ratio=0.15):
    """
    Temporal split of windowed data.
    Uses the LAST val_ratio portion as validation.
    """
    n = len(X)
    n_val = int(n * val_ratio)

    X_train = X[:-n_val]
    y_train = y[:-n_val]

    X_val = X[-n_val:]
    y_val = y[-n_val:]

    return X_train, y_train, X_val, y_val


# ---------------- Model ----------------
def load_model(input_shape, loss_fn):
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(1e-3),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    model.summary()
    return model


# ---------------- Training ----------------
def train(model, X_train, y_train, X_val, y_val, model_path, class_weights=None, epochs=50, batch_size=16, callbacks=None):
    callbacks = callbacks or []
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        verbose=0,
        callbacks=[TqdmCallback(verbose=1)] + callbacks
    )
    model.save(model_path)
    print(f"Model saved in {model_path}")
    return history



# ---------------- Evaluation ----------------
def save_classification_results_json(y_test, y_pred, filepath="results.json"):
    cm = confusion_matrix(y_test, y_pred).tolist()
    cr = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    results = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm,
        "classification_report": cr
    }

    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results in {filepath}")



def eval(model, X_test, y_test, save_path="results.json"):
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    save_classification_results_json(y_test, y_pred, filepath=save_path)
    return y_pred



# Loss re-definition -- Focal Loss
def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        # p_t
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)

        # focal loss
        loss = -alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(loss)

    return loss



def class_weights_loss(class_weights):
    """
    Simple cost-sensitive binary cross-entropy.
    class_weights = {0: w0, 1: w1}
    """
    w0 = tf.constant(class_weights[0], dtype=tf.float32)
    w1 = tf.constant(class_weights[1], dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        weights = tf.where(
            tf.equal(y_true, 1.0),
            w1,
            w0
        )

        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(weights * bce)

    return loss



class CostSensitiveLambda(tf.keras.callbacks.Callback):
    def __init__(self, ir_overall, beta=0.9, lambda_min=1.0, lambda_max=None):
        super().__init__()
        self.ir = tf.constant(ir_overall, tf.float32)
        self.beta = beta
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_major = tf.Variable(ir_overall, trainable=False, dtype=tf.float32)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        acc = tf.constant(logs.get("accuracy", 0.0), tf.float32)
        gmean = tf.constant(logs.get("gmean", 0.0), tf.float32)

        lam_raw = self.ir * tf.exp(-gmean) * tf.exp(-acc)

        # EMA smoothing
        lam = self.beta * self.lambda_major + (1 - self.beta) * lam_raw

        # Optional bounds (soft safety)
        lam = tf.maximum(lam, self.lambda_min)
        if self.lambda_max is not None:
            lam = tf.minimum(lam, self.lambda_max)

        self.lambda_major.assign(lam)


def cost_sensitive_loss(lambda_var):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        weights = tf.where(tf.equal(y_true, 0.0), lambda_var, 1.0)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(weights * bce)
    return loss
