import numpy as np
import pandas as pd

nInst = 50
currentPos = np.zeros(nInst)
clusters = [[19, 32, 48], [22, 46, 43, 37], [41, 35], [20, 25, 26]]  # clusters from analysis


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

def getMyPosition(prices):
    nInst, _ = prices.shape
    positions = np.zeros(nInst)
    
    # Calculate MACD, RSI, and returns
    macd, _ = calculate_macd(pd.DataFrame(prices.T))
    rsi = calculate_rsi(pd.DataFrame(prices.T))
    returns = calculate_returns(pd.DataFrame(prices.T))
    
    # Use provided clusters to determine positions
    for cluster in clusters:
        cluster_macd = macd.iloc[-1, cluster].mean()
        cluster_rsi = rsi.iloc[-1, cluster].mean()
        
        # Simple strategy: If MACD > 0 and RSI < 70, go long; if MACD < 0 and RSI > 30, go short
        if cluster_macd > 0 and cluster_rsi < 70:
            for stock in cluster:
                positions[stock] = 10000 / prices[stock, -1]  # Long position
        elif cluster_macd < 0 and cluster_rsi > 30:
            for stock in cluster:
                positions[stock] = -10000 / prices[stock, -1]  # Short position
    
    # Ensure position limits
    for i in range(nInst):
        current_price = prices[i, -1]
        position_value = positions[i] * current_price
        if abs(position_value) > 10000:
            positions[i] = 10000 / current_price if position_value > 0 else -10000 / current_price
    
    return positions.astype(int)
