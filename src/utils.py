import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
class LogHGR(HistGradientBoostingRegressor):
    """Gradient boosting on log(target): suits heavy-tailed viscosity-driven indices."""
    def fit(self, X, y, **k): return super().fit(X, np.log(np.maximum(y, 1e-3)), **k)
    def predict(self, X): return np.exp(super().predict(X))
