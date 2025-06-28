import numpy as np
from sklearn.linear_model import LinearRegression  # Add this import
from sklearn.preprocessing import PolynomialFeatures  # Add this import

class QuadraticRegressionModel:
    def __init__(self, degree=2, lookback=5):
        self.degree = degree
        self.lookback = lookback
        self.poly = PolynomialFeatures(degree=self.degree)
        self.model = LinearRegression()
    
    def fit(self, prices):
        """
        Train the model on historical prices for a single instrument.
        :param prices: 1D array of historical prices in chronological order
        """
        # Compute daily returns
        returns = np.diff(prices) / prices[:-1]
        
        n = len(returns)
        if n < self.lookback + 1:
            raise ValueError(f"Need ≥ {self.lookback+2} days of prices. Got {len(prices)+1}.")
        
        # Create feature matrix X and target vector y
        X = np.array([returns[i - self.lookback:i] 
                      for i in range(self.lookback, n)])
        y = returns[self.lookback:]  # Target: next day's return
        
        # Generate polynomial features and train model
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
    
    def predict_next(self, recent_prices):
        """
        Predict next day's return using the most recent prices.
        :param recent_prices: Array of last (lookback + 1) days' prices
        :return: Predicted return for the next day
        """
        if len(recent_prices) < self.lookback + 1:
            raise ValueError(f"Need ≥ {self.lookback+1} recent prices. Got {len(recent_prices)}.")
        
        # Compute returns from recent prices
        recent_returns = np.diff(recent_prices) / recent_prices[:-1]
        X_recent = recent_returns[-self.lookback:].reshape(1, -1)
        
        # Transform features and predict
        X_recent_poly = self.poly.transform(X_recent)
        return self.model.predict(X_recent_poly)[0]