import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# check for missing values
def check_missing_values(df):
    print(df.isnull().sum())


# handle missing values
def handle_missing_values(df, method='ffill'):
    df = df.fillna(method=method)
    return df


# identify outliers
def identify_outliers(df, threshold=3):
    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < threshold).all(axis=1)
    new_df = df[filtered_entries]
    return new_df


# plot time series
def plot_time_series(df):
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(df.index, df['value'])
    ax.set(xlabel='Date', ylabel='Value', title='Time Series Plot')
    plt.show()


# plot histogram
def plot_histogram(df):
    fig, ax = plt.subplots(figsize=(15,5))
    ax.hist(df['value'], bins=30)
    ax.set(xlabel='Value', ylabel='Frequency', title='Histogram Plot')
    plt.show()


# group by hourly and daily and add results to original df
def add_hourly_and_daily_to_dataframe(df):
    # resample data to hourly frequency and add to DataFrame
    df['hourly'] = df['data'].resample('H').mean().values

    # resample data to daily frequency and add to DataFrame
    df['daily'] = df['data'].resample('D').mean().values

    return df


# resample data to hourly frequency and add to DataFrame
def add_hourly_to_dataframe(df):
    df['hourly'] = df['data'].resample('H').mean().values

    return df


# resample data to daily frequency and add to DataFrame
def add_daily_to_dataframe(df):
    df['daily'] = df['data'].resample('D').mean().values

    return df
