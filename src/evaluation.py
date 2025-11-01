import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)

def load_models(models_path='models/'):
    """Load all saved model pickle files"""
    models = {}
    for filename in os.listdir(models_path):
        if filename.endswith('.pkl') and filename not in ['scaler.pkl', 'test_data.pkl']:
            model_name = filename.replace('.pkl', '').replace('_', ' ').title()
            with open(os.path.join(models_path, filename), 'rb') as f:
                models[model_name] = pickle.load(f)
    return models

def load_test_data(models_path='models/'):
    """Load test data and scaler"""
    with open(os.path.join(models_path, 'test_data.pkl'), 'rb') as f:
        X_test, y_test = pickle.load(f)
    with open(os.path.join(models_path, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return X_test, y_test, scaler

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics and ROC data"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    return accuracy, report, roc_auc, cm, fpr, tpr

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, model_name):
    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

def main():
    print("\n🚀 Starting Model Evaluation...\n")
    models = load_models()
    X_test, y_test, scaler = load_test_data()
    
    roc_scores = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        accuracy, report, roc_auc, cm, fpr, tpr = evaluate_model(model, X_test, y_test)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("Classification Report:\n", report)
        
        plot_confusion_matrix(cm, name)
        plot_roc_curve(fpr, tpr, roc_auc, name)
        
        roc_scores[name] = roc_auc
    
    # Summary of ROC AUC scores
    print("\nSummary of ROC AUC scores:")
    for model_name, auc_score in roc_scores.items():
        print(f"{model_name}: {auc_score:.4f}")

if __name__ == "__main__":
    main()
