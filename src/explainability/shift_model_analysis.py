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
    Return model predictions (binary) and probabilities (that lead to the predictions using the threshold)
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



# Quantitative evaluation

def evaluate_warning_on_fn(stats_df, fn_mask):
    """
    Measures how many False Negatives fall inside early warning regimes.
    """
    fn_df = stats_df.loc[fn_mask]

    covered = (fn_df["y_early"] == 1).sum()
    total = len(fn_df)

    return {
        "n_false_negatives": total,
        "fn_with_early_warning": covered,
        "coverage_rate": covered / total if total > 0 else np.nan
    }

















# def plot_global_shift_with_early_warning(
#     stats_df,
#     threshold=None,
#     title="Global Distribution Shift with Early Warning Regimes"
# ):
#     fig, ax = plt.subplots(figsize=(13, 4))

#     # Linha do shift
#     ax.plot(
#         stats_df.index,
#         stats_df["shift_global"],
#         label="Distribution shift (global)",
#         linewidth=2
#     )

#     # Threshold (opcional, apenas como referência)
#     if threshold is not None:
#         ax.axhline(
#             threshold,
#             color="orange",
#             linestyle="--",
#             linewidth=2,
#             label="Reference threshold"
#         )

#     # Early warning regimes (ANTES da crise)
#     early = stats_df["y_early"] == 1
#     ax.fill_between(
#         stats_df.index,
#         stats_df["shift_global"].min(),
#         stats_df["shift_global"].max(),
#         where=early,
#         color="orange",
#         alpha=0.25,
#         label="Early warning regime"
#     )

#     # Crises reais
#     crisis = stats_df[stats_df["y_true"] == 1]
#     ax.scatter(
#         crisis.index,
#         crisis["shift_global"],
#         color="red",
#         s=40,
#         label="Crisis",
#         zorder=5
#     )

#     ax.set_ylabel("Normalized Wasserstein shift")
#     ax.set_xlabel("Date")
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(alpha=0.3)

#     plt.tight_layout()
#     plt.show()

# def compute_global_shift(window, baseline_data, normalize=True):
#     if normalize:
#         w, b = normalize_using_baseline(window, baseline_data)
#     else:
#         w, b = window, baseline_data

#     shifts = [
#         wasserstein_distance(w[:, j], b[:, j])
#         for j in range(w.shape[1])
#     ]
#     return float(np.mean(shifts))

# def compute_feature_shift(window, baseline_data, feature_names, normalize=True):
#     if normalize:
#         w, b = normalize_using_baseline(window, baseline_data)
#     else:
#         w, b = window, baseline_data

#     return {
#         name: float(wasserstein_distance(w[:, j], b[:, j]))
#         for j, name in enumerate(feature_names)
#     }

# def compute_shift_over_time(X, y_true, y_early, dates, baseline_data, feature_names, normalize=True, compute_feature_shifts=True):
#     rows_global = []
#     rows_feat = []

#     for i, window in enumerate(X):
#         global_shift = compute_global_shift(window, baseline_data, normalize)
#         rows_global.append({
#             "date": dates[i],
#             "y_true": int(y_true[i]),
#             "y_early":int(y_early[i]),
#             "shift_global": global_shift
#         })

#         if compute_feature_shifts:
#             d = {
#                 "date": dates[i],
#                 "y_true": int(y_true[i]),
#                 "y_early":int(y_early[i]),
#             }
#             d.update(compute_feature_shift(window, baseline_data, feature_names, normalize))
#             rows_feat.append(d)

#     stats_df = pd.DataFrame(rows_global).set_index("date")
#     feat_df = pd.DataFrame(rows_feat).set_index("date") if rows_feat else None
#     return stats_df, feat_df


# def error_type_series(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> np.ndarray:
#     """
#     Produz um array com: TP, FP, FN, TN (por janela).
#     """
#     out = np.empty(len(y_true), dtype=object)

#     for i, (t, p) in enumerate(zip(y_true, y_pred)):
#         if t == positive_label and p == positive_label:
#             out[i] = "TP"
#         elif t != positive_label and p == positive_label:
#             out[i] = "FP"
#         elif t == positive_label and p != positive_label:
#             out[i] = "FN"
#         else:
#             out[i] = "TN"

#     return out


# def build_behavior_df(model_name: str,
#                       dates: pd.DatetimeIndex,
#                       y_true: np.ndarray,
#                       y_pred: np.ndarray,
#                       y_proba: np.ndarray,
#                       stats_df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Junta:
#       - y_true
#       - y_pred
#       - y_proba
#       - error_type (TN/FP/FN/TP)
#       - shift_global (vindo do stats_df)

#     stats_df tem de ter index=date e coluna shift_global.
#     """
#     behavior_df = pd.DataFrame({
#         "date": dates,
#         "model": model_name,
#         "y_true": y_true.astype(int),
#         "y_pred": y_pred.astype(int),
#         "proba": y_proba.astype(float),
#     }).set_index("date")

#     behavior_df["error_type"] = error_type_series(
#         behavior_df["y_true"].values,
#         behavior_df["y_pred"].values
#     )

#     # join do shift_global
#     behavior_df = behavior_df.join(stats_df[["shift_global"]], how="left")

#     return behavior_df


# def make_early_labels(y, horizon):
#     y_early = np.zeros_like(y)
#     for t in range(len(y) - horizon):
#         y_early[t] = int(np.any(y[t+1 : t+1+horizon] == 1))
#     return y_early

# def raise_warning(shift_value, threshold):
#     return shift_value > threshold


# def plot_global_shift(stats_df, threshold=None):
#     plt.figure(figsize=(13, 4))
#     stats_df["shift_global"].plot(label="Distribution shift")

#     if threshold is not None:
#         plt.axhline(threshold, linestyle="--", color="gray", label="Warning threshold")

#     crisis = stats_df[stats_df["y_true"] == 1]
#     plt.scatter(crisis.index, crisis["shift_global"], color="red", s=30, label="Crisis")

#     plt.title("Global Distribution Shift over Time")
#     plt.ylabel("Normalized Wasserstein shift")
#     plt.xlabel("Date")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()


# def plot_shift_with_fn(stats_df, fn_mask):
#     plt.figure(figsize=(13, 4))
#     stats_df["shift_global"].plot(label="Distribution shift")

#     fn = stats_df.loc[fn_mask]
#     plt.scatter(fn.index, fn["shift_global"], marker="x", color="red", label="False Negatives")

#     plt.title("Shift when model predicts NON-crisis (False Negatives)")
#     plt.ylabel("Shift (global)")
#     plt.xlabel("Date")
#     plt.legend()
#     plt.grid(alpha=0.3)
#     plt.tight_layout()
#     plt.show()



# def plot_shift_by_error_type(behavior_df):
#     order = ["TN", "FP", "FN", "TP"]
#     data = [
#         behavior_df.loc[behavior_df["error_type"] == k, "shift_global"].dropna()
#         for k in order
#     ]

#     plt.figure(figsize=(7, 4))
#     plt.boxplot(data, labels=order, showfliers=False)
#     plt.ylabel("Global shift")
#     plt.title("Shift distribution by model error type")
#     plt.grid(axis="y", alpha=0.3)
#     plt.tight_layout()
#     plt.show()


# def plot_top_features_for_error(feature_shift_df, behavior_df,
#                                 feature_names, error_type="FN", top_k=10):
#     idx = behavior_df.index[behavior_df["error_type"] == error_type]
#     sub = feature_shift_df.loc[feature_shift_df.index.intersection(idx), feature_names]

#     mean_shift = sub.mean().sort_values(ascending=False).head(top_k)

#     mean_shift.sort_values().plot.barh(figsize=(7, 4))
#     plt.title(f"Top features driving {error_type} errors")
#     plt.xlabel("Average normalized Wasserstein shift")
#     plt.tight_layout()
#     plt.show()

#     return mean_shift



# # Goal: obtain the stress windows (the ones that appear highlighted) and check if they appear before crisis in the normal dataset. 

# def apply_warning_system_model_predictions():
#     # get_model_fn_errors() --> it will give the 2020 crisis period
#     # check (numbers, visualizations) if we could have triggered a crisis warning based on this strategy
#     pass