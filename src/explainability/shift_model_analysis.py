import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


def window_dates_from_index(index, n_steps):
    '''
    Aligns each window X[i] with the corresponding prediction date.
    '''
    return pd.DatetimeIndex(index[n_steps:])

# Models and Predictions
def get_model_predictions(model, X, threshold=0.6):
    '''
    Returns model predictions (binary) and probabilities (that lead to the predictions using the threshold)
    '''
    y_proba = model.predict(X, verbose=0).reshape(-1)
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred, y_proba


def get_false_negatives(y_true, y_pred):
    '''
    False Negatives = Crisis exists (1) but the model predicted Normal (0)
    '''
    # print("Y_TRUE: ", y_true)
    return (y_true == 1) & (y_pred == 0)


# Baseline and Normlization
def make_baseline_data(X, y, normal_label=0):
    '''
    Flattens all non-crisis windows into baseline distribution.
    '''
    X0 = X[y == normal_label]
    return X0.reshape(-1, X.shape[-1])


def normalize_using_baseline(window, baseline_data, eps=1e-8):
    '''
    Normalization using baseline statistics (mean and standard deviation)
    '''
    mu = baseline_data.mean(axis=0)
    sd = baseline_data.std(axis=0) + eps
    return (window - mu) / sd, (baseline_data - mu) / sd


# Distributions Shift
def compute_global_shift(window, baseline_data, normalize=True):
    '''
    Mean Wasserstein distance across features.
    '''
    if normalize:
        w, b = normalize_using_baseline(window, baseline_data)
    else:
        w, b = window, baseline_data

    return float(np.mean([
        wasserstein_distance(w[:, j], b[:, j])
        for j in range(w.shape[1])
    ]))

# Early warning labels
def make_early_labels(y, horizon):
    '''
    y_early[t]=1, if a crisis occurs within next 'horizon' steps.
    '''
    y_early = np.zeros_like(y)
    for t in range(len(y) - horizon):
        y_early[t] = int(np.any(y[t+1:t+1+horizon] == 1))
    return y_early


# shifts over time
def compute_shift_over_time(X, y_true, y_early, dates, baseline_data):
    """
    Computes global distribution shift for each window.
    """
    rows = []

    for i, window in enumerate(X):
        rows.append({
            "date": dates[i],
            "y_true": int(y_true[i]),
            "y_early": int(y_early[i]),
            "shift_global": compute_global_shift(window, baseline_data)
        })

    return pd.DataFrame(rows).set_index("date")


# PLOT - Early Warning and Crisis
def plot_global_shift_with_early_warning(stats_df, title=None):

    fig, ax = plt.subplots(figsize=(13, 4))

    ax.plot(stats_df.index, stats_df["shift_global"], linewidth=2, label="Distribution shift (global)")

    # Early warning regimes
    early = stats_df["y_early"] == 1
    ax.fill_between(stats_df.index, stats_df["shift_global"].min(), stats_df["shift_global"].max(), where=early,
                    color="orange", alpha=0.25, label="Early warning regime (horizon)")

    # Real crises
    crisis = stats_df[stats_df["y_true"] == 1]
    ax.scatter(crisis.index, crisis["shift_global"], color="red", s=40, label="Crisis", zorder=5)

    ax.set_ylabel("Normalized Wasserstein shift")
    ax.set_xlabel("Date")
    ax.set_title(title or "Global Distribution Shift with Early Warning Regimes")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# PLOT - False Negatives  vs.  Early Warning

def plot_fn_with_early_warning(stats_df, fn_mask, title=None):
    """
    Shows whether statistical stress anticipates crises missed by the model.
    """
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(stats_df.index, stats_df["shift_global"], linewidth=2, label="Distribution shift (global)")

    # Early warning regimes
    early = stats_df["y_early"] == 1
    ax.fill_between(stats_df.index, stats_df["shift_global"].min(), stats_df["shift_global"].max(), where=early,
                    color="orange", alpha=0.25, label="Early warning regime")

    # False Negatives
    fn = stats_df.loc[fn_mask]
    
    ax.scatter(fn.index, fn["shift_global"], marker="x", color="red", s=80, label="False Negatives (missed crises)", zorder=5)

    ax.set_title(title or "False Negatives vs Statistical Early Warning")
    ax.set_ylabel("Normalized Wasserstein shift")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
