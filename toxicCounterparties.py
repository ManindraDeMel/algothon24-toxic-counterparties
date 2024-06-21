
import numpy as np

nInst = 50
currentPos = np.zeros(nInst)


# Define the getMyPosition function
def getMyPosition(prices, model_path='model.pkl', scaler_path='scaler.pkl'):
    nInst, nt = prices.shape
    
    # Load the model and scaler
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    features = generate_features(prices)
    latest_features = features[:, -1, :]
    latest_features_scaled = scaler.transform(latest_features)
    
    # Predict the position changes
    position_changes = model.predict(latest_features_scaled)
    
    # Calculate the desired positions
    current_prices = prices[:, -1]
    max_position = 10000 // current_prices
    positions = max_position * position_changes
    
    return positions

