"""
Generates 3 realistic sample CSV datasets.
Skips files that already exist.
Run: python data/generate_samples.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'samples')
os.makedirs(SAMPLES_DIR, exist_ok=True)


def generate_credit_fraud():
    path = os.path.join(SAMPLES_DIR, 'credit_fraud.csv')
    if os.path.exists(path):
        print(f"Skipping {path} (already exists)")
        return
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=7,
        n_redundant=2, weights=[0.95, 0.05], flip_y=0.05, random_state=42,
    )
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'transaction_amount':   np.abs(X[:,0] * 800 + 250).round(2),
        'account_age_days':     np.abs(X[:,1] * 500 + 800).clip(1, 3650).astype(int),
        'n_transactions_today': np.abs(X[:,2] * 4 + 5).clip(0, 30).astype(int),
        'merchant_risk_score':  (X[:,3] * 0.2 + 0.5).clip(0, 1).round(3),
        'hour_of_day':          rng.integers(0, 24, 2000),
        'is_international':     (X[:,4] > 0).astype(int),
        'device_trust_score':   (X[:,5] * 0.15 + 0.75).clip(0, 1).round(3),
        'velocity_score':       (X[:,6] * 0.2 + 0.4).clip(0, 1).round(3),
        'ip_risk':              (X[:,7] * 0.15 + 0.2).clip(0, 1).round(3),
        'time_since_last_txn':  np.abs(X[:,8] * 3600 + 7200).astype(int),
        'fraud':                y,
    })
    df.to_csv(path, index=False)
    print(f"Generated {path}  ({len(df)} rows)")


def generate_employee_churn():
    path = os.path.join(SAMPLES_DIR, 'employee_churn.csv')
    if os.path.exists(path):
        print(f"Skipping {path} (already exists)")
        return
    X, y = make_classification(
        n_samples=1500, n_features=9, n_informative=7,
        n_redundant=1, weights=[0.80, 0.20], random_state=7,
    )
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        'age':                   (X[:,0] * 8 + 38).clip(22, 65).round(0).astype(int),
        'tenure_years':          np.abs(X[:,1] * 3 + 5).clip(0, 30).round(1),
        'salary':                (X[:,2] * 1500 + 5500).clip(2000, 15000).round(0).astype(int),
        'department':            rng.choice(['Engineering','Sales','HR','Finance','Marketing'], 1500),
        'performance_score':     (X[:,3] * 1.2 + 3.0).clip(1, 5).round(1),
        'n_projects':            np.abs(X[:,4] * 2 + 4).clip(1, 12).astype(int),
        'overtime_hours':        np.abs(X[:,5] * 8 + 10).clip(0, 40).round(0).astype(int),
        'satisfaction_score':    (X[:,6] * 1.5 + 3.5).clip(1, 5).round(1),
        'last_promotion_years':  np.abs(X[:,7] * 1.5 + 2).clip(0, 10).round(0).astype(int),
        'distance_from_office':  np.abs(X[:,8] * 10 + 15).clip(1, 60).round(1),
        'churn':                 y,
    })
    df.to_csv(path, index=False)
    print(f"Generated {path}  ({len(df)} rows)")


def generate_house_prices():
    path = os.path.join(SAMPLES_DIR, 'house_prices.csv')
    if os.path.exists(path):
        print(f"Skipping {path} (already exists)")
        return
    X, y = make_regression(
        n_samples=1000, n_features=10, n_informative=7,
        noise=20000, random_state=42,
    )
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'area_sqft':           (X[:,0] * 600 + 1800).clip(400, 8000).round(0).astype(int),
        'bedrooms':            (X[:,1] * 1.5 + 3).clip(1, 8).round(0).astype(int),
        'bathrooms':           (X[:,2] * 1.0 + 2).clip(1, 5).round(1),
        'age_years':           np.abs(X[:,3] * 15 + 20).clip(0, 100).round(0).astype(int),
        'garage':              (X[:,4] > 0).astype(int),
        'neighborhood_score':  (X[:,5] * 1.5 + 5).clip(1, 10).round(1),
        'school_rating':       (X[:,6] * 1.2 + 6).clip(1, 10).round(1),
        'distance_to_city_km': np.abs(X[:,7] * 8 + 12).clip(1, 50).round(1),
        'renovation_year':     rng.integers(1990, 2024, 1000),
        'floor_level':         rng.integers(1, 20, 1000),
        'price':               (y * 60 + 320000).clip(80000, 1500000).round(0).astype(int),
    })
    df.to_csv(path, index=False)
    print(f"Generated {path}  ({len(df)} rows)")


if __name__ == '__main__':
    generate_credit_fraud()
    generate_employee_churn()
    generate_house_prices()
    print("All sample datasets ready.")
