import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import STL
from statsmodels.stats.diagnostic import normal_ad
from statsmodels.graphics.tsaplots import plot_acf

from textblob import TextBlob


# perform STL decomposition
def perform_seasonal_decomposition(df, seasonal_window=13, plot=True):
    stl = STL(df['value'], seasonal=seasonal_window)
    res = stl.fit()

    if plot:
        # plot STL decomposition
        fig, ax = plt.subplots(figsize=(15,8))
        ax.plot(df.index, res.observed, label='Observed')
        ax.plot(df.index, res.seasonal, label='Seasonal')
        ax.plot(df.index, res.trend, label='Trend')
        ax.plot(df.index, res.resid, label='Residual')
        ax.set(xlabel='Date', title='STL Decomposition')
        ax.legend()
        plt.show()

        # plot residuals
        residuals = res.resid.dropna()
        residuals.plot.hist(bins=30)
        plt.show()

        # plot autocorrelation of residuals
        plot_acf(residuals, lags=30)
        plt.show()

    # perform normality test on residuals
    _, p = normal_ad(residuals)
    if p < 0.05:
        print("Residuals are not normally distributed")
    else:
        print("Residuals are normally distributed")

    return res


# perform descriptive analysis
def perform_descriptive_analysis(df):
    print(df.describe())

    return df


# perform sentiment analysis
def perform_sentiment_analysis(df):
    df['sentiment_score'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment'] = df['sentiment_score'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')

    # plot sentiment over time
    fig, ax = plt.subplots(figsize=(15,8))
    sns.lineplot(x=df.index, y='sentiment_score', data=df)
    ax.set(xlabel='Date', ylabel='Sentiment Score', title='Sentiment Analysis')
    plt.show()

    # print sentiment distribution
    print("Sentiment distribution:")
    print(df['sentiment'].value_counts())

    return df


# calculate rolling statistics
def rolling_statistics(df, window_size):
    # Calculate rolling mean and standard deviation with specified window size
    rolling_mean = df.rolling(window=window_size).mean()
    rolling_std = df.rolling(window=window_size).std()

    # Add rolling mean and standard deviation to DataFrame
    df['rolling_mean'] = rolling_mean.values
    df['rolling_std'] = rolling_std.values

    return df


# plot original data and rolling mean and standard deviation
def plot_rolling_statistics(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['data'], label='Original Data')
    plt.plot(df.index, df['rolling_mean'], label='Rolling Mean')
    plt.plot(df.index, df['rolling_std'], label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show()
