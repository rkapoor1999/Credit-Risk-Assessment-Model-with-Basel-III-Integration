import numpy as np
import pandas as pd

def assign_basel_risk_weight(pd_value):
    """
    Assign risk weight based on Basel III standards
    
    Parameters:
    -----------
    pd_value : float
        Probability of default
        
    Returns:
    --------
    float
        Risk weight (as a decimal)
    """
    # Simplified risk weight assignment based on Basel III
    if pd_value <= 0.05:
        return 0.5  # 50% risk weight
    elif pd_value <= 0.10:
        return 0.75  # 75% risk weight
    elif pd_value <= 0.30:
        return 1.0  # 100% risk weight
    else:
        return 1.5  # 150% risk weight

def calculate_pd(model, data):
    """
    Calculate probability of default
    
    Parameters:
    -----------
    model : object
        Trained model with predict_proba method
    data : pandas.DataFrame
        Data to calculate PD for
        
    Returns:
    --------
    numpy.ndarray
        Probability of default
    """
    return model.predict_proba(data)[:, 1]

def estimate_lgd(loan_data):
    """
    Estimate loss given default
    
    Parameters:
    -----------
    loan_data : pandas.DataFrame
        Loan data
        
    Returns:
    --------
    numpy.ndarray
        Loss given default
    """
    # Basic LGD logic - in reality would be a trained model
    # Higher FICO score = lower LGD
    base_lgd = 0.45  # Industry average
    
    # Adjust based on collateral
    if 'home_ownership_OWN' in loan_data.columns:
        lgd = np.where(loan_data['home_ownership_OWN'] == 1, 
                      base_lgd * 0.8,  # Lower LGD for homeowners
                      base_lgd)
    else:
        lgd = base_lgd
        
    return lgd

def calculate_ead(loan_data, original_data):
    """
    Calculate exposure at default
    
    Parameters:
    -----------
    loan_data : pandas.DataFrame
        Processed loan data
    original_data : pandas.DataFrame
        Original loan data with loan amounts
        
    Returns:
    --------
    numpy.ndarray
        Exposure at default
    """
    # Match indices between datasets
    loan_amounts = original_data.loc[loan_data.index, 'loan_amnt']
    return loan_amounts

def calculate_rwa(ead, risk_weight):
    """
    Calculate risk-weighted assets
    
    Parameters:
    -----------
    ead : numpy.ndarray
        Exposure at default
    risk_weight : numpy.ndarray
        Risk weight
        
    Returns:
    --------
    numpy.ndarray
        Risk-weighted assets
    """
    return ead * risk_weight