import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)
clusters = [[41,35]]#[[19, 32, 48], [22, 46, 43, 37], [41, 35], [20, 25, 26]]  # clusters from analysis

def calculate_bollinger_bands(prices, window=20, num_std_dev=2):
    sma = prices.rolling(window=window).mean()
    std_dev = prices.rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return sma, upper_band, lower_band

def calculate_macd(prices, slow=26, fast=12, signal=9):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_returns(prices):
    returns = prices.pct_change()
    return returns

def calculate_volatility(prices, window=14):
    return prices.pct_change().rolling(window=window).std()


def getMyPosition(prices):
    nInst, nt = prices.shape
    positions = np.zeros(nInst)
    
    # Calculate indicators
    macd, _ = calculate_macd(pd.DataFrame(prices.T))
    rsi = calculate_rsi(pd.DataFrame(prices.T))
    sma, upper_band, lower_band = calculate_bollinger_bands(pd.DataFrame(prices.T))
    volatility = calculate_volatility(pd.DataFrame(prices.T))
    
    # Use provided clusters to determine positions
    for cluster in clusters:
        cluster_macd = macd.iloc[-1, cluster].mean()
        cluster_rsi = rsi.iloc[-1, cluster].mean()
        cluster_volatility = volatility.iloc[-1, cluster].mean()
        cluster_upper_band = upper_band.iloc[-1, cluster].mean()
        cluster_lower_band = lower_band.iloc[-1, cluster].mean()
        
        # Define position sizing based on volatility (inverse)
        position_size = 10000 / (cluster_volatility + 1e-6)  # Adding small value to avoid division by zero
        
        # Advanced strategy: Use multiple indicators and dynamic position sizing
        for stock in cluster:
            current_price = prices[stock, -1]
            if current_price < cluster_lower_band and cluster_macd > 0 and cluster_rsi < 70:
                for stock in cluster:
                    positions[stock] = position_size / prices[stock, -1]  # Long position
            elif current_price > cluster_upper_band and cluster_macd < 0 and cluster_rsi > 30:
                for stock in cluster:
                    positions[stock] = -position_size / prices[stock, -1]  # Short position
    
    # Ensure position limits
    for i in range(nInst):
        current_price = prices[i, -1]
        position_value = positions[i] * current_price
        if abs(position_value) > 10000:
            positions[i] = 10000 / current_price if position_value > 0 else -10000 / current_price
    
    return positions.astype(int)