from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np



# --------------------------------------- Dataset Overview and Description ---------------------------------------

def df_info(df):
    '''
    Plots a dataframe with information regarding the original df columns. Includes: 
        - Data type
        - Missing values
        - Unique values
        - Statistics: mean, standard deviation and Min/Max values

    Args:
        df (pd.DataFrame): Input dataframe to summarize.
    Returns:
        pd.DataFrame: Summary dataframe with the information described above.
    '''
    summary = pd.DataFrame(index=df.columns)
    float_cols = df.select_dtypes(include=['float']).columns
    summary['dtype'] = df.dtypes
    summary['missing'] = df.isnull().sum()
    summary['n_unique'] = df.nunique()
    summary['mean'] = df[float_cols].mean()
    summary['std'] = df[float_cols].std()
    summary['min'] = df[float_cols].min()
    summary['max'] = df[float_cols].max()
    return summary

# --------------------------------------- Time Series Analysis ---------------------------------------

def convert_date_to_datetime(df, date_column_name):
    '''
    Converts the date column to *datetime* type and replaces 'date_column_name' by 'date', removing the original column. 
    Sets the df_index as date!
    '''
    df = df.copy()
    df['date'] = pd.to_datetime(df[date_column_name], errors='coerce')
    if date_column_name != 'date':
        df = df.drop(columns=[date_column_name])
    df = df.set_index('date')
    return df


def verify_ts_time_frequency(df, name):
    ''' 
    Verifies the time frequency of a given time series. 
    Return the df name and the time frequency.
    '''
    if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise TypeError(f"DataFrame {name} index is not datetime type")
    freq = pd.infer_freq(df.index)
    return name, freq


def aggregate_trimesters(ts_df, ts_df_name, time_series_freq, aggregate_method, target_frequency='QS-OCT'):
    '''
    Aggregates (if necessary) the input time series into the target time frequency, to further merge datasets.
    We can select different aggregation methods: mean, sum or median.
    Returns the aggregated dataframe.
    '''
    curr_df = ts_df.copy()
    if time_series_freq != target_frequency:
        if aggregate_method == 'mean':
            curr_df = curr_df.resample(target_frequency).mean()
        elif aggregate_method == 'sum':
            curr_df = curr_df.resample(target_frequency).sum()
        elif aggregate_method == 'median':
            curr_df = curr_df.resample(target_frequency).median()
        else:
            raise ValueError("agg_method must be 'mean', 'sum' or 'median'")
        
        print(f"{ts_df_name} Time Series frequency changed from {time_series_freq} to {pd.infer_freq(curr_df.index)}")
        print("Aggregated dataset:\n", curr_df.head())
    else: 
        print(f"{ts_df_name} Time Series is already in the target frequency ({target_frequency})")

    return curr_df



def plot_time_series(df, column_name):
    '''
    Plots the time series according to the dataframe index (date).
    '''
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise TypeError(f"Index is not datetime type.")

    df = df.sort_index()  
 
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 4))
    
    plt.plot(df.index, df[column_name], color="#AD93C3", linewidth=1.5, label=column_name)

    plt.title(f"{column_name} Time Series", fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel(column_name, fontsize=12)

    locator = mdates.AutoDateLocator(minticks=10, maxticks=20)
    plt.gca().xaxis.set_major_locator(locator)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.gcf().autofmt_xdate()

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.show()


def run_adf_test(df, column_name):
    '''
    Runs the ADF test to check time series stationarity. Returns a Dataframe with the test results.
    '''
    results = []
    series = df[column_name].dropna()
    result = adfuller(series)
    test_stat = result[0]
    p_value = result[1]
    crit_values = result[4]

    results.append({
        'Feature': column_name,
        'ADF Statistic': test_stat,
        'p-value': p_value,
        'Stationary': p_value < 0.05
    })
    results_df = pd.DataFrame(results)

    return results_df, (p_value < 0.05) # returns df and True if stationary, False otherwise


def difference_ts(df, column_name):
    '''
    Log-differences the input time series
    '''
    diff_column_name = f"{column_name}_diff"
    df[diff_column_name] = df[column_name].diff()
    return df, diff_column_name


def plot_correlations(df, column_name, lags):
    '''
    Plots ACF and PACF to check the most relevant lag for further construction of time windows (Important for modelling)
    '''
    series = df[column_name].dropna()
    n = len(series)
    
    acf_vals = acf(series, nlags=lags)
    pacf_vals = pacf(series, nlags=lags, method='ywm')
    
    conf = 1.96/np.sqrt(n) 
    
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    
    # --- ACF ---
    for i, val in enumerate(acf_vals):
        axes[0].vlines(i, 0, val, color="#88AFCE")
    axes[0].hlines([conf, -conf], xmin=0, xmax=lags, colors='r', linestyles='dashed', label='Significance')
    axes[0].hlines(0, xmin=0, xmax=lags, colors='black')
    axes[0].set_title(f'ACF - {column_name}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Lag')
    axes[0].set_ylabel('ACF')
    axes[0].legend()
    
    # --- PACF ---
    for i, val in enumerate(pacf_vals):
        axes[1].vlines(i, 0, val, color="#81A759")
    axes[1].hlines([conf, -conf], xmin=0, xmax=lags, colors='r', linestyles='dashed', label='Significance')
    axes[1].hlines(0, xmin=0, xmax=lags, colors='black')
    axes[1].set_title(f'PACF - {column_name}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Lag')
    axes[1].set_ylabel('PACF')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()



def time_series_analysis_main(df, date_column_name, column_name, lags=24, aggregate_method='mean'):
    # 1. df_info
    info = df_info(df)
    print('Dataset Information:\n', info)

    # 2. convert date to datetime
    df_ts = convert_date_to_datetime(df, date_column_name) # df_ts is now the dataset with index as date
    print('Time Series Dataset:\n', df_ts.head())    

    name, frequency = verify_ts_time_frequency(df_ts, column_name)
    df_ts = aggregate_trimesters(df_ts, name, frequency, aggregate_method)

    print()

    # 3. plot time series
    plot_time_series(df_ts, column_name=column_name)

    # 4. run ADF test for stationarity check
    adf_df, is_stationary = run_adf_test(df_ts, column_name=column_name)
    print("ADF test results:\n", adf_df)

    # 5. difference the time series, until reaching stationarity
    df_current = df_ts.copy()
    current_col = column_name
    difference_order = 0
    while not is_stationary: # Time series not stationary

        df_current, current_col = difference_ts(df_current, column_name=current_col)
        difference_order += 1
        print("Differenced Time Series df:\n", df_current)

        adf_df, is_stationary = run_adf_test(df_current, column_name=current_col)
        print("ADF test results:\n", adf_df)

        plot_time_series(df_current, column_name=current_col)
    
    if difference_order > 0:
        print(f"The Time Series is now stationary, after {difference_order} differencing")

    plot_correlations(df_current, current_col, lags)
    return df_current


# --------------------------------------- Data Merging ---------------------------------------

def merge_data_with_label(labels_df, list_all_df, list_all_df_names):
    
    merged_df = labels_df.copy()
    for df, name in zip(list_all_df, list_all_df_names):
        temp_df = df.copy()
        
        if temp_df.index.name != 'date':
            temp_df = temp_df.reset_index().rename(columns={temp_df.index.name: 'date'})

        merged_df = pd.merge(merged_df, temp_df, on='date', how='left', suffixes=('', f'_{name}'))
    
    return merged_df