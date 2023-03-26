from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd


def preprocess_categorical(data, categorical_col):
    """
    Preprocess time series data with a categorical column.

    Parameters:
        data (pandas.DataFrame): Time series data to preprocess.
        categorical_col (str): Name of the categorical column.

    Returns:
        preprocessed_data (pandas.DataFrame): Preprocessed time series data.
    """
    # Create a copy of the data to avoid modifying the original DataFrame
    preprocessed_data = data.copy()

    # Convert categorical column to numerical using LabelEncoder
    le = LabelEncoder()
    preprocessed_data[categorical_col] = le.fit_transform(preprocessed_data[categorical_col])

    # Create dummy variables using OneHotEncoder
    ohe = OneHotEncoder(sparse=False)
    dummy_vars = ohe.fit_transform(preprocessed_data[[categorical_col]])
    dummy_vars_df = pd.DataFrame(dummy_vars, columns=[f"{categorical_col}_{i}" for i in range(dummy_vars.shape[1])])

    # Concatenate the dummy variables with the original data and drop the original categorical column
    preprocessed_data = pd.concat([preprocessed_data, dummy_vars_df], axis=1)
    preprocessed_data = preprocessed_data.drop(columns=[categorical_col])

    return preprocessed_data


def split_data(data, target, test_size):
    """
    Split time series data into training and testing sets.

    Parameters:
        data (numpy.ndarray): Time series data to split.
        target (numpy.ndarray): Target variable for time series data.
        test_size (float): Proportion of data to use for testing (e.g., 0.2 for 20% testing data).

    Returns:
        X_train (numpy.ndarray): Training input data.
        X_test (numpy.ndarray): Testing input data.
        y_train (numpy.ndarray): Training target data.
        y_test (numpy.ndarray): Testing target data.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test


def scale_data(train_data, test_data):
    """
    Scale time series data using MinMaxScaler after splitting into training and testing sets.

    Parameters:
        train_data (numpy.ndarray): Training data.
        test_data (numpy.ndarray): Testing data.

    Returns:
        scaled_train_data (numpy.ndarray): Scaled training data.
        scaled_test_data (numpy.ndarray): Scaled testing data.
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler object used to scale the data.
    """
    # Create scaler object
    scaler = MinMaxScaler()

    # Fit scaler on training data
    scaler.fit(train_data)

    # Transform training and testing data using scaler
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    return scaled_train_data, scaled_test_data, scaler


def preprocess_data(data, target, test_size=0.2):
    """
    Split time series data into training and testing sets and perform feature scaling.

    Parameters:
        data (numpy.ndarray): Time series data to preprocess.
        target (numpy.ndarray): Target variable for time series data.
        test_size (float): Proportion of data to use for testing (e.g., 0.2 for 20% testing data).

    Returns:
        X_train_scaled (numpy.ndarray): Scaled training input data.
        X_test_scaled (numpy.ndarray): Scaled testing input data.
        y_train (numpy.ndarray): Training target data.
        y_test (numpy.ndarray): Testing target data.
        scaler (sklearn.preprocessing.MinMaxScaler): Scaler object used to scale the data.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data, target, test_size)

    # Create scaler object
    scaler = MinMaxScaler()

    # Fit scaler on training data
    scaler.fit(X_train)

    # Transform training and testing data using scaler
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

