import pandas as pd

from src.analysis.eda import *
from src.analysis.statistics import *
from src.analysis.transform import *
from src.models.lstm_model import *

# load data
df = pd.read_csv('your_data_file.csv', parse_dates=['date_col'], index_col='date_col')

# check for missing values
check_missing_values(df)

# handle missing values
df = handle_missing_values(df)

# identify outliers
df_outliers = identify_outliers(df)

# plot time series
plot_time_series(df)

# plot histogram
plot_histogram(df)

# perform seasonal decomposition
res = perform_seasonal_decomposition(df)

# perform descriptive analysis
df = perform_descriptive_analysis(df)

# perform sentiment analysis
df = perform_sentiment_analysis(df)

# group data by hourly and daily frequency and add the results to the original DataFrame
df = add_hourly_and_daily_to_dataframe(df)

# group data by hourly and add the results to the original DataFrame
df = add_hourly_to_dataframe(df)

# group data by daily frequency and add the results to the original DataFrame
df = add_daily_to_dataframe(df)

# calculate the rolling mean and standard deviation using a window size of 24 hours
df = rolling_statistics(df, window_size=24)

# plot original data and rolling mean and standard deviation
df = rolling_statistics(df, window_size=24)
plot_rolling_statistics(df)

# handling categorical values
preprocessed_data = preprocess_categorical(data, categorical_col)

# split time series data into training and testing sets, and apply feature scaling
X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(data, target, test_size)

# calculate the accuracy, confusion matrix, precision and f1-score
accuracy, cm, precision, f1 = lstm_model(X_train_scaled, y_train, X_test_scaled, y_test)
