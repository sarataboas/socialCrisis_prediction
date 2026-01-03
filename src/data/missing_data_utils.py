import matplotlib.pyplot as plt
import pandas as pd


# ------------------------ Missing Data Analysis ------------------------

def compute_feature_coverage(df):
    coverage = []
    for col in df.columns:
        if df[col].isna().all():
            continue
        start = df[col].first_valid_index()
        end = df[col].last_valid_index()
        start_pos = df.index.get_loc(start)
        end_pos = df.index.get_loc(end)

        n_missing = df[col].isna().sum()
        pct_missing = df[col].isna().mean()

        coverage_days = None
        if isinstance(df.index, pd.DatetimeIndex):
            coverage_days = (end - start).days

        coverage.append({
            "feature": col,
            # "start_index": start,
            # "end_index": end,
            "start_pos": start_pos,
            "end_pos": end_pos,
            "start_date": start if isinstance(start, pd.Timestamp) else None,
            "end_date": end if isinstance(end, pd.Timestamp) else None,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "coverage_days": coverage_days
        })

    coverage_df = pd.DataFrame(coverage).sort_values("start_date").reset_index(drop=True)
    return coverage_df


def plot_missing_by_decade(df, cmap="viridis", figsize=(8, 6), title="Missingness by Variable and Decade"):

    df_tmp = df.copy()
    df_tmp["decade"] = (df_tmp.index.year // 10) * 10

    missing_by_decade = (
        df_tmp
        .groupby("decade")
        .apply(lambda x: x.isna().mean())
        .T
    )
    plt.figure(figsize=figsize)
    plt.imshow(missing_by_decade, aspect="auto", cmap=cmap)
    plt.colorbar(label="% Missing")
    plt.yticks(range(len(missing_by_decade.index)), missing_by_decade.index)
    plt.xticks(range(len(missing_by_decade.columns)), missing_by_decade.columns, rotation=45)
    plt.title(title)
    plt.xlabel("Decade")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.show()

    return missing_by_decade 



# -------------------------------- Handle Missing Data --------------------------------

def prefix_impute_first_value_with_flag(df):
    df_out = df.copy()
    # print(df_out.columns)
    columns = [col for col in df.columns if col != 'label']

    for col in columns:
        s = df_out[col]

        df_out[f"{col}_missing"] = s.isna().astype(int)

        if not s.isna().any():
            continue

        first_valid = s.first_valid_index()
        if first_valid is None: # never happens but just to be sure
            continue

        first_val = s.loc[first_valid]

        prefix_mask = df_out.index < first_valid
        df_out.loc[prefix_mask, col] = df_out.loc[prefix_mask, col].fillna(first_val)
        df_out.loc[~prefix_mask, col] = df_out.loc[~prefix_mask, col].ffill()

    return df_out
