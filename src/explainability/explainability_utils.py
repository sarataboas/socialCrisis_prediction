import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

import seaborn as sns
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer


# --------------------- Integrated Gradients ---------------------

def plot_ig_heatmap(ig, feature_names, title, cmap="RdYlGn_r", figsize=(12, 4.5), show_present_line=True):
    if ig.ndim == 3:
        ig = ig[0]

    assert ig.shape[1] == len(feature_names), \
        "Number of features does not match ig shape"

    vmax = np.max(np.abs(ig))

    plt.figure(figsize=figsize)

    ax = sns.heatmap(ig, cmap=cmap, vmin=-vmax, vmax=vmax, center=0, linewidths=0.3, linecolor="white", cbar_kws={"label": "Integrated Gradient"})

    ax.set_yticks(np.arange(ig.shape[0]) + 0.5)
    ax.set_yticklabels([f"t-{i}" for i in range(ig.shape[0]-1, -1, -1)], fontsize=9 )

    # eixo X → features
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)

    ax.set_ylabel("Time step (past → recent)", fontsize=10)
    ax.set_xlabel("Feature", fontsize=10)

    ax.set_title(title, fontsize=12, pad=10)

    # linha a marcar o presente (t-0)
    if show_present_line:
        ax.axhline( y=ig.shape[0]-0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_ig_feature_importance(ig, feature_names, title="Feature Importance (Integrated Gradients)", figsize=(6, 3), color="crimson"):
    if ig.ndim == 3:
        ig = ig[0]

    assert ig.shape[1] == len(feature_names), \
        "Number of features does not match IG shape"

    importance = np.sum(np.abs(ig), axis=0)
    order = np.argsort(importance)[::-1]
    importance_sorted = importance[order]
    feature_names_sorted = [feature_names[i] for i in order]

    plt.figure(figsize=figsize)
    sns.barplot(x=importance_sorted, y=feature_names_sorted, color=color, edgecolor=color, alpha=0.85)

    plt.xlabel("Σ |Integrated Gradient| over time")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()




# --------------------- Occlusion ---------------------

def predict_proba(model, X):
    return model.predict(X, verbose=0).reshape(-1)


def occlusion_feature_time(model, x_instance, baseline="zero"):
    assert x_instance.ndim == 3 and x_instance.shape[0] == 1
    T, F = x_instance.shape[1], x_instance.shape[2]

    base_pred = predict_proba(model, x_instance)[0]
    occlusion_map = np.zeros((T, F))

    if baseline == "zero":
        baseline_values = np.zeros((T, F))
    elif baseline == "mean":
        baseline_values = np.mean(x_instance[0], axis=0, keepdims=True)
        baseline_values = np.repeat(baseline_values, T, axis=0)
    else:
        raise ValueError("baseline must be 'zero' or 'mean'")

    for t in range(T):
        for f in range(F):
            x_occ = x_instance.copy()
            x_occ[0, t, f] = baseline_values[t, f]

            pred_occ = predict_proba(model, x_occ)[0]
            occlusion_map[t, f] = base_pred - pred_occ

    return occlusion_map




def plot_occlusion_heatmap(occlusion_map, feature_names, title, cmap="RdBu"):
    vmax = np.max(np.abs(occlusion_map))

    plt.figure(figsize=(12, 4.5))
    ax = sns.heatmap(occlusion_map, cmap=cmap, vmin=-vmax, vmax=vmax, center=0,linewidths=0.3, linecolor="white", cbar_kws={"label": "Δ Prediction (Occlusion)"})
    ax.set_yticks(np.arange(occlusion_map.shape[0]) + 0.5)
    ax.set_yticklabels(
        [f"t-{i}" for i in range(occlusion_map.shape[0]-1, -1, -1)],
        fontsize=9
    )

    # x-axis: features
    ax.set_xticks(np.arange(len(feature_names)) + 0.5)
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=9)

    ax.set_ylabel("Time step (past → recent)")
    ax.set_xlabel("Feature")

    ax.set_title(title, fontsize=12, pad=10)

    # linha do presente (t-0)
    ax.axhline(y=occlusion_map.shape[0]-0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_occlusion_feature_importance(occlusion_map, feature_names, title="Feature Importance (Occlusion)"):
  
    importance = np.sum(np.abs(occlusion_map), axis=0)
    df_imp = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importance
        })
        .sort_values("importance", ascending=False)
    )
    plt.figure(figsize=(6, 3))
    sns.barplot(data=df_imp, x="importance", y="feature", color="#7178B4FF")

    plt.xlabel("Σ |Δ Prediction| over time")
    plt.ylabel("Feature")
    plt.title(title)

    plt.tight_layout()
    plt.show()



    # ---------------------------- LIME ---------------------------

def lime_predict_fn(X_flat, X_test, model):
    """
    X_flat: (N, T*F)
    returns: (N, 2) probs for [class 0, class 1]
    """
    X = X_flat.reshape((-1, X_test.shape[1], X_test.shape[2]))
    p1 = model.predict(X, verbose=0).reshape(-1)
    return np.vstack([1 - p1, p1]).T


def lime_to_dataframe(lime_exp, label=1):
    rows = lime_exp.as_list(label=label)

    df = pd.DataFrame(rows, columns=["feature", "weight"])
    df["abs_weight"] = df["weight"].abs()

    return df.sort_values("abs_weight", ascending=False)


def plot_lime_surrogate(
    df_lime,
    title="LIME — Local surrogate explanation",
    max_features=20,
    figsize=(8, 5)
):
    df = df_lime.head(max_features).copy()

    df["effect"] = df["weight"].apply(
        lambda x: "Increases crisis risk" if x > 0 else "Mitigates crisis risk"
    )

    plt.figure(figsize=figsize)

    sns.barplot(
        data=df,
        x="weight",
        y="feature",
        hue="effect",
        palette={
            "Increases crisis risk": "#EF575E",  
            "Mitigates crisis risk": "#BACC62"    
        },
        dodge=False
    )

    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Surrogate weight")
    plt.ylabel("Feature @ time")
    plt.title(title)
    plt.legend(title="")
    plt.tight_layout()
    plt.show()



def plot_lime_by_feature(df_lime, figsize=(8, 4)):
    df = df_lime.copy()
    df["base_feature"] = df["feature"].str.split("@").str[0]

    agg = (
        df.groupby("base_feature")["weight"]
        .apply(lambda x: np.sum(np.abs(x)))
        .sort_values(ascending=False)
        .reset_index()
    )

    plt.figure(figsize=figsize)
    sns.barplot(
        data=agg,
        x="weight",
        y="base_feature",
        color="#73A56C"
    )

    plt.xlabel("Σ |Surrogate weight| over time")
    plt.ylabel("Feature")
    plt.title("LIME — Feature importance (time-aggregated)")
    plt.tight_layout()
    plt.show()
