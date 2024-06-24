import numpy as np
from sklearn.linear_model import LinearRegression

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    
    newPos = np.zeros(nins)

    for i in range(nins):
        prices = prcSoFar[i, :]
        times = np.arange(nt).reshape(-1, 1)
        
        # Fit linear regression model
        model = LinearRegression().fit(times, prices)
        trend = model.predict(times)
        
        # Calculate standard deviation of residuals
        residuals = prices - trend
        std_dev = np.std(residuals)
        
        # Calculate upper and lower bounds
        upper_bound = trend + 1*std_dev
        lower_bound = trend - 1*std_dev
        
        # Determine the position based on crossing the bounds
        if prices[-1] > upper_bound[-1]:
            newPos[i] = -5000 / prices[-1]  # Sell
        elif prices[-1] < lower_bound[-1]:
            newPos[i] = 5000 / prices[-1]  # Buy
        else:
            newPos[i] = 0  # Hold

    currentPos = currentPos + newPos
    currentPos = np.round(currentPos).astype(int)
    return currentPos
