import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_data(filepath='data/creditcard.csv'):
    """Load the credit card dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nClass distribution:")
        print(df['Class'].value_counts())
        print(f"\nFraud percentage: {(df['Class'].sum() / len(df)) * 100:.2f}%")
        return df
    except FileNotFoundError:
        print(f"❌ Error: File not found at {filepath}")
        print("Please download the dataset from Kaggle and place it in the data/ folder")
        return None

def explore_data(df):
    """Basic data exploration"""
    print("\n" + "="*70)
    print("DATA EXPLORATION")
    print("="*70)
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nMissing Values:")
    print(df.isnull().sum().sum())
    
    print("\nStatistical Summary:")
    print(df.describe())
    
    return df

def preprocess_data(df):
    """Preprocess and split the data"""
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✅ Data split completed!")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale the 'Amount' and 'Time' features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[['Amount', 'Time']] = scaler.fit_transform(X_train[['Amount', 'Time']])
    X_test_scaled[['Amount', 'Time']] = scaler.transform(X_test[['Amount', 'Time']])
    
    print(f"\n✅ Feature scaling completed!")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def handle_imbalance(X_train, y_train):
    """Apply SMOTE to handle class imbalance"""
    print("\n" + "="*70)
    print("HANDLING CLASS IMBALANCE WITH SMOTE")
    print("="*70)
    
    print(f"\nBefore SMOTE:")
    print(f"Class 0 (Legitimate): {(y_train == 0).sum()}")
    print(f"Class 1 (Fraud): {(y_train == 1).sum()}")
    
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"\nAfter SMOTE:")
    print(f"Class 0 (Legitimate): {(y_train_balanced == 0).sum()}")
    print(f"Class 1 (Fraud): {(y_train_balanced == 1).sum()}")
    print(f"✅ Classes are now balanced!")
    
    return X_train_balanced, y_train_balanced

if __name__ == "__main__":
    print("\n🚀 Starting Data Preprocessing Pipeline...\n")
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Explore data
        explore_data(df)
        
        # Preprocess
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        
        # Handle imbalance
        X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)
        
        print("\n✅ Data preprocessing completed successfully!")
        print("\nNext step: Run model_training.py to train the models")
