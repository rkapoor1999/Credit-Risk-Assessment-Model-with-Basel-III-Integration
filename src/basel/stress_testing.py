import numpy as np
import pandas as pd

def apply_stress_scenario(base_pd, scenario='moderate'):
    """
    Apply stress scenario to probability of default
    
    Parameters:
    -----------
    base_pd : numpy.ndarray
        Base probability of default
    scenario : str
        Stress scenario ('mild', 'moderate', 'severe')
        
    Returns:
    --------
    numpy.ndarray
        Stressed probability of default
    """
    scenarios = {
        'mild': 1.5,       # 50% increase in defaults
        'moderate': 2.0,   # Double defaults
        'severe': 3.0      # Triple defaults
    }
    
    multiplier = scenarios.get(scenario, 1.0)
    stressed_pd = base_pd * multiplier
    
    # Cap at 1.0
    return np.minimum(stressed_pd, 1.0)

def calculate_stress_metrics(loan_data, base_pd, lgd, ead, scenario='moderate'):
    """
    Calculate stress metrics
    
    Parameters:
    -----------
    loan_data : pandas.DataFrame
        Loan data
    base_pd : numpy.ndarray
        Base probability of default
    lgd : numpy.ndarray
        Loss given default
    ead : numpy.ndarray
        Exposure at default
    scenario : str
        Stress scenario ('mild', 'moderate', 'severe')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with stress metrics
    """
    from ..basel.risk_weights import assign_basel_risk_weight
    
    # Apply stress scenario
    stressed_pd = apply_stress_scenario(base_pd, scenario)
    
    # Calculate stressed metrics
    stressed_el = stressed_pd * lgd * ead
    stressed_risk_weight = np.array([assign_basel_risk_weight(pd) for pd in stressed_pd])
    stressed_rwa = ead * stressed_risk_weight
    
    # Create results DataFrame
    results = loan_data.copy()
    results['Stressed_PD'] = stressed_pd
    results['Stressed_EL'] = stressed_el
    results['Stressed_RiskWeight'] = stressed_risk_weight
    results['Stressed_RWA'] = stressed_rwa
    
    return results

def summarize_stress_results(normal_metrics, stressed_metrics):
    """
    Summarize stress test results
    
    Parameters:
    -----------
    normal_metrics : dict
        Dictionary with normal metrics
    stressed_metrics : dict
        Dictionary with stressed metrics
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with stress test summary
    """
    metrics = ['Total Exposure', 'RWA', 'Tier 1 Capital', 
              'Total Capital', 'With Buffer']
    
    normal_values = [
        normal_metrics['exposure'],
        normal_metrics['rwa'],
        normal_metrics['tier1_capital'],
        normal_metrics['total_capital'],
        normal_metrics['capital_with_buffer']
    ]
    
    stressed_values = [
        stressed_metrics['exposure'],
        stressed_metrics['rwa'],
        stressed_metrics['tier1_capital'],
        stressed_metrics['total_capital'],
        stressed_metrics['capital_with_buffer']
    ]
    
    results = pd.DataFrame({
        'Metric': metrics,
        'Normal': normal_values,
        'Stressed': stressed_values,
        'Change (%)': [(s - n) / n * 100 for n, s in zip(normal_values, stressed_values)]
    })
    
    return results