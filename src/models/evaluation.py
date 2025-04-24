from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    Parameters:
    -----------
    model : object
        Trained model with predict and predict_proba methods
    X_test : pandas.DataFrame
        Testing feature matrix
    y_test : pandas.Series
        Testing target variable
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': (y_pred == y_test).mean(),
        'auc': roc_auc_score(y_test, y_pred_prob),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }
    
    return metrics

def plot_roc_curve(y_test, y_pred_prob, figsize=(10, 6)):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    y_test : pandas.Series
        Testing target variable
    y_pred_prob : numpy.ndarray
        Predicted probabilities
    figsize : tuple
        Figure size
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc