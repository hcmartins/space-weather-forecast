import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import LSTM, Dense

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score


def build_lstm_model(X_train, y_train, num_units=50, num_epochs=100, batch_size=32):
    # Reshape input data for LSTM layer
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=num_units, input_shape=(1, X_train.shape[2])))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train LSTM model
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size)

    print(model.summary())

    return model


# evaluate the predictions using accuracy, confusion matrix, precision, and F1-score
def evaluate_lstm_model(model, X_test, y_test):
    # Reshape test input data for LSTM layer
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Make predictions on test data using LSTM model
    y_pred = model.predict(X_test)

    # Round predicted values to nearest integer
    y_pred = np.round(y_pred)

    # Evaluate predictions using accuracy, confusion matrix, precision, and F1-score
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, confusion, precision, f1


# evaluate and plot the predictions
def evaluate_and_plot_lstm_model(model, X_test, y_test):
    # Reshape test input data for LSTM layer
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Make predictions on test data using LSTM model
    y_pred = model.predict(X_test)

    # Round predicted values to nearest integer
    y_pred = np.round(y_pred)

    # Evaluate predictions using accuracy, confusion matrix, precision, and F1-score
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Plot confusion matrix
    sns.set(font_scale=1.4) # Adjust to fit labels
    sns.heatmap(confusion, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion matrix')
    plt.show()

    return accuracy, confusion, precision, f1


# generic
def lstm_model(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Train and evaluate an LSTM model on scaled time series data.

    Parameters:
        X_train_scaled (numpy.ndarray): Scaled training input data.
        y_train (numpy.ndarray): Training target data.
        X_test_scaled (numpy.ndarray): Scaled testing input data.
        y_test (numpy.ndarray): Testing target data.

    Returns:
        accuracy (float): Accuracy score of the model.
        cm (numpy.ndarray): Confusion matrix of the model.
        precision (float): Precision score of the model.
        f1 (float): F1 score of the model.
    """
    # Reshape data for LSTM model
    X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(1, X_train_scaled.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train LSTM model
    model.fit(X_train_scaled, y_train, epochs=50, verbose=0)

    # Make predictions on testing data
    y_pred = model.predict(X_test_scaled)
    y_pred = y_pred.reshape(y_test.shape)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred.round())
    cm = confusion_matrix(y_test, y_pred.round())
    precision = precision_score(y_test, y_pred.round())
    f1 = f1_score(y_test, y_pred.round())

    return accuracy, cm, precision, f1
