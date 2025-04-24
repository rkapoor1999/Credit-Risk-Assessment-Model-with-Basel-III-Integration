from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_logistic_regression(X_train, y_train, max_iter=1000):
    """
    Train logistic regression model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : pandas.Series
        Training target variable
    max_iter : int
        Maximum number of iterations
        
    Returns:
    --------
    sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=max_iter)
    log_reg.fit(X_train, y_train)
    return log_reg

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train random forest model
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : pandas.Series
        Training target variable
    n_estimators : int
        Number of trees in the forest
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    sklearn.ensemble.RandomForestClassifier
        Trained random forest model
    """
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    return rf_model

def get_feature_importance(model, feature_names):
    """
    Get feature importance from random forest model
    
    Parameters:
    -----------
    model : sklearn.ensemble.RandomForestClassifier
        Trained random forest model
    feature_names : list
        List of feature names
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with feature names and importance scores
    """
    import pandas as pd
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return feature_importance

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)