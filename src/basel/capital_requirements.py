def calculate_minimum_capital(rwa, tier1_ratio=0.06, total_ratio=0.08, conservation_buffer=0.025):
    """
    Calculate minimum capital requirements based on Basel III
    
    Parameters:
    -----------
    rwa : float
        Total risk-weighted assets
    tier1_ratio : float
        Tier 1 capital ratio requirement
    total_ratio : float
        Total capital ratio requirement
    conservation_buffer : float
        Capital conservation buffer requirement
        
    Returns:
    --------
    dict
        Dictionary with capital requirements
    """
    tier1_capital = rwa * tier1_ratio
    total_capital = rwa * total_ratio
    capital_with_buffer = rwa * (total_ratio + conservation_buffer)
    
    return {
        'tier1_capital': tier1_capital,
        'total_capital': total_capital,
        'capital_with_buffer': capital_with_buffer
    }

def calculate_capital_ratios(capital, rwa):
    """
    Calculate capital ratios
    
    Parameters:
    -----------
    capital : float
        Available capital
    rwa : float
        Total risk-weighted assets
        
    Returns:
    --------
    float
        Capital ratio
    """
    return capital / rwa

def check_capital_adequacy(available_capital, required_capital):
    """
    Check if capital is adequate
    
    Parameters:
    -----------
    available_capital : float
        Available capital
    required_capital : float
        Required capital
        
    Returns:
    --------
    bool
        True if capital is adequate, False otherwise
    """
    return available_capital >= required_capital