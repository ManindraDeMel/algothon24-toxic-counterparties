
import numpy as np

nInst = 50
currentPos = np.zeros(nInst)


def getMyPosition(prcSoFar):
    global currentPos
    (nInst, nt) = prcSoFar.shape

    if nt < 20:
        # Not enough data for meaningful prediction
        return np.zeros(nInst)

    # Calculate log returns for the past 20 days
    returns = np.log(prcSoFar[:, 1:] / prcSoFar[:, :-1])
    recent_returns = returns[:, -20:]

    # Calculate moving averages
    short_window = 5
    long_window = 20
    short_ma = np.mean(prcSoFar[:, -short_window:], axis=1)
    long_ma = np.mean(prcSoFar[:, -long_window:], axis=1)

    # Generate signals: 1 for buy, -1 for sell, 0 for hold
    signals = np.zeros(nInst)
    signals[short_ma > long_ma] = 1
    signals[short_ma < long_ma] = -1

    # Determine positions based on signals
    position_values = signals * 5000 / prcSoFar[:, -1]
    
    # Clip positions to comply with $10k limit
    position_limits = 10000 / prcSoFar[:, -1]
    clipped_positions = np.clip(position_values, -position_limits, position_limits)

    newPos = clipped_positions.astype(int)
    deltaPos = newPos - currentPos
    currentPos = newPos

    return currentPos

