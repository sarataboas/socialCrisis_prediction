import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.utils import resample

import warnings
warnings.filterwarnings("ignore")

# --------------------- Check for Imbalance ----------------------

# 1. Measures
def compute_imbalance_measures(df, target_column_name):
    '''    
    Computes the imbalance measures - Imbalance Ratio and Normalized Imbalance Measure
    '''
    class_counts = df[target_column_name].value_counts()
    n_majority = class_counts.max()
    n_minority = class_counts.min()

    ir = n_majority / n_minority
    c2 = 1 - (1 / ir)

    print(f"Class counts:\n{class_counts}\n")
    print(f"Imbalance Ratio (IR): {ir:.3f}")
    print(f"Normalized Imbalance Measure (C2): {c2:.3f}")


# 2. Plot
def plot_target_distribution(df, target_column_name):
    '''
    Plots the target class distribution.
    '''
    colors = ["#89CCCF", "#E8A0A0"]
    sns.set(style="whitegrid", palette=colors)
    counts = df[target_column_name].value_counts()

    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        dodge=False,
        palette=colors
    )

    plt.title('Crisis Label Class Distribution', fontsize=10, fontweight='bold')
    plt.xlabel('Label', fontsize=10)
    plt.ylabel('Number of Records', fontsize=10)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend([], [], frameon=False)  # Remove legenda redundante

    # Ajustar margens e mostrar
    plt.tight_layout()
    plt.show()


# ------------------- Window-based Event Oversampling -------------------

def create_windows_for_oversampling(df, n_steps, target_column):
    feature_cols = [c for c in df.columns if c != target_column]

    X, y = [], []
    data = df[feature_cols + [target_column]].values

    for i in range(len(data) - n_steps+1):
        # window of features 
        X.append(data[i:i+n_steps, :-1])

        # window label 
        window_label = data[i:i+n_steps, -1].max()
        y.append(window_label)
        
    return np.array(X), np.array(y)


def oversample_windows(X, y, random_state=42):
    # split classes
    X_majority = X[y==0]
    X_minority = X[y == 1]

    y_majority = y[y == 0]
    y_minority = y[y == 1]

    # oversample minority windows
    X_minority_resampled, y_minority_resampled = resample(X_minority, y_minority, replace=True, n_samples=len(y_majority), random_state=random_state)

    X_balanced = np.concatenate([X_majority, X_minority_resampled])
    y_balanced = np.concatenate([y_majority, y_minority_resampled])

    return X_balanced, y_balanced


def windows_to_dataframe(X, y, feature_cols, n_steps, target_column='label'):
    """
    Converts windowed data (X, y) into a flat DataFrame.
    """
    n_windows, _, n_features = X.shape

    columns = [f"{feat}_t{t}" for t in range(n_steps) for feat in feature_cols]

    # flatten windows
    X_flat = X.reshape(n_windows, n_steps * n_features)

    df_windows = pd.DataFrame(X_flat, columns=columns)
    df_windows[target_column] = y

    return df_windows



# ------------------- Train History Plots -------------------

def plot_training_history(history, title="Training evolution"):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Accuracy & G-mean
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="val accuracy", linestyle="--")

    if "gmean" in history.history:
        plt.plot(history.history["gmean"], label="train gmean")
        plt.plot(history.history["val_gmean"], label="val gmean", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
