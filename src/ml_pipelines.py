"""
Script to store ML-pipelines used for experiments.
"""

import numpy as np

from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

alphas = np.logspace(-10, 10, num=100)
models = {
    "baseline" : DummyRegressor(),
    "basic_linear_model" : Pipeline([
        ('scaler',StandardScaler()),
        ('regression',LinearRegression())]),
    "ridge" : Pipeline([
        ('scaler',StandardScaler()),
        ('regression',RidgeCV(alphas=alphas, store_cv_results=True))]),
    "ridge_spline_nystroem" : Pipeline([
        ('scaler',StandardScaler()),
        ('spline',SplineTransformer(degree=100,n_knots=5)),
        ('nystroem',Nystroem(kernel='rbf',n_components=80,gamma=0.025,random_state=42)),
        ('regression',RidgeCV(alphas=alphas, store_cv_results=True))]),
    "hist_gradient_boosting_regressor" : HistGradientBoostingRegressor(),
    "random_forest" : RandomForestRegressor()
    }
