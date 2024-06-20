import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Calculate RSI
def calculate_rsi(prices, window=14):
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100. / (1. + rs)
    
    for i in range(window, len(prices)):
        delta = deltas[i - 1]

        if delta > 0:
            up_val = delta
            down_val = 0.
        else:
            up_val = 0.
            down_val = -delta

        up = (up * (window - 1) + up_val) / window
        down = (down * (window - 1) + down_val) / window

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

# Calculate MACD
def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Feature engineering
def generate_features(prices):
    features = []
    for i in range(prices.shape[0]):
        instrument_prices = prices[i]
        returns = np.diff(instrument_prices) / instrument_prices[:-1]
        rsi = calculate_rsi(instrument_prices)
        macd, signal = calculate_macd(pd.Series(instrument_prices))
        
        # Combine features
        instrument_features = np.vstack([returns, rsi[1:], macd[1:], signal[1:]]).T
        features.append(instrument_features)
    
    # Make sure all features are of the same length
    min_length = min(f.shape[0] for f in features)
    features = np.array([f[-min_length:] for f in features])
    
    return features

# Model training and saving
def train_and_save_model(prices, model_path='model.pkl', scaler_path='scaler.pkl'):
    features = generate_features(prices)
    positions = np.zeros_like(prices)  # Initial positions for training
    
    X = features.reshape(-1, features.shape[2])
    y = np.sign(positions[:, 1:].flatten())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model and scaler
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    
    with open(scaler_path, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

if __name__ == "__main__":
    prices = np.loadtxt('prices.txt')
    train_and_save_model(prices)  # Train the model and save it
