import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Skipping XGBoost model.")

from data_preprocessing import load_data, preprocess_data, handle_imbalance

def train_models(X_train, y_train):
    """Train multiple classification models"""
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    # Add XGBoost only if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"✅ {name} training completed!")
    
    return trained_models

def save_models(models, scaler, X_test, y_test, filepath='models/'):
    """Save trained models and test data to disk"""
    print("\n" + "="*70)
    print("SAVING MODELS AND TEST DATA")
    print("="*70)
    
    os.makedirs(filepath, exist_ok=True)
    
    # Save each model
    for name, model in models.items():
        filename = filepath + name.replace(' ', '_').lower() + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Saved {name} to {filename}")
    
    # Save scaler
    with open(filepath + 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Saved scaler to {filepath}scaler.pkl")
    
    # Save test data for evaluation
    with open(filepath + 'test_data.pkl', 'wb') as f:
        pickle.dump((X_test, y_test), f)
    print(f"✅ Saved test data to {filepath}test_data.pkl")

if __name__ == "__main__":
    print("\n🚀 Starting Model Training Pipeline...\n")
    
    # Load and preprocess data
    df = load_data()
    
    if df is not None:
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        # Handle imbalance
        X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)
        
        # Train models
        models = train_models(X_train_balanced, y_train_balanced)
        
        # Save models and test data
        save_models(models, scaler, X_test, y_test)
        
        print("\n" + "="*70)
        print("✅ Model training completed successfully!")
        print("="*70)
        print("\nNext step: Run evaluation.py to evaluate the models")
