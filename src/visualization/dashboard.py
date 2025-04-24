import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_pd_distribution(pd_values, title='Distribution of Probability of Default (PD)', nbins=50):
    """
    Plot distribution of probability of default
    
    Parameters:
    -----------
    pd_values : numpy.ndarray
        Probability of default values
    title : str
        Plot title
    nbins : int
        Number of bins
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    fig = px.histogram(pd_values, nbins=nbins, title=title)
    fig.update_layout(
        xaxis_title='Probability of Default',
        yaxis_title='Count'
    )
    return fig

def compare_pd_distributions(normal_pd, stressed_pd, title='Normal vs Stressed PD Comparison', nbins=50):
    """
    Compare normal and stressed PD distributions
    
    Parameters:
    -----------
    normal_pd : numpy.ndarray
        Normal probability of default values
    stressed_pd : numpy.ndarray
        Stressed probability of default values
    title : str
        Plot title
    nbins : int
        Number of bins
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=normal_pd, name='Normal PD', opacity=0.7, nbinsx=nbins))
    fig.add_trace(go.Histogram(x=stressed_pd, name='Stressed PD', opacity=0.7, nbinsx=nbins))
    fig.update_layout(
        title=title,
        xaxis_title='Probability of Default',
        yaxis_title='Count',
        barmode='overlay'
    )
    return fig

def plot_risk_weight_distribution(risk_weights, ead, title='Distribution of Exposure by Risk Weight'):
    """
    Plot distribution of exposure by risk weight
    
    Parameters:
    -----------
    risk_weights : numpy.ndarray
        Risk weight values
    ead : numpy.ndarray
        Exposure at default values
    title : str
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    # Create a DataFrame with risk weights and EAD
    risk_df = pd.DataFrame({
        'RiskWeight': risk_weights,
        'EAD': ead
    })
    
    # Sum EAD by risk weight
    risk_summary = risk_df.groupby('RiskWeight')['EAD'].sum().reset_index()
    
    # Create pie chart
    fig = px.pie(risk_summary, values='EAD', names='RiskWeight', 
                title=title,
                labels={'RiskWeight': 'Risk Weight', 'EAD': 'Exposure at Default'})
    
    return fig

def plot_expected_loss_by_risk(pd_values, el_values, title='Expected Loss by Risk Category'):
    """
    Plot expected loss by risk category
    
    Parameters:
    -----------
    pd_values : numpy.ndarray
        Probability of default values
    el_values : numpy.ndarray
        Expected loss values
    title : str
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    # Create risk categories based on PD values
    risk_df = pd.DataFrame({
        'PD': pd_values,
        'EL': el_values
    })
    
    risk_df['RiskCategory'] = pd.cut(risk_df['PD'], 
                                     bins=[0, 0.05, 0.1, 0.3, 1.0],
                                     labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Sum EL by risk category
    risk_summary = risk_df.groupby('RiskCategory')['EL'].sum().reset_index()
    
    # Create bar chart
    fig = px.bar(risk_summary, x='RiskCategory', y='EL',
                title=title,
                color='RiskCategory',
                labels={'RiskCategory': 'Risk Category', 'EL': 'Expected Loss'})
    
    return fig

def plot_capital_requirements(capital_data, title='Capital Requirements by Scenario'):
    """
    Plot capital requirements by scenario
    
    Parameters:
    -----------
    capital_data : pandas.DataFrame
        DataFrame with capital requirements by scenario
    title : str
        Plot title
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure
    """
    # Ensure capital_data has the right format
    expected_columns = ['Scenario', 'Tier 1 Capital', 'Additional Capital', 'Conservation Buffer']
    if not all(col in capital_data.columns for col in expected_columns):
        raise ValueError(f"capital_data must have columns: {expected_columns}")
    
    # Create stacked bar chart
    fig = px.bar(capital_data, x='Scenario', 
                y=['Tier 1 Capital', 'Additional Capital', 'Conservation Buffer'],
                title=title, 
                barmode='stack',
                labels={'value': 'Capital Amount', 'variable': 'Capital Component'})
    
    return fig

def create_risk_metrics_table(loans_df, pd_col='PD', el_col='EL', rwa_col='RWA', ead_col='EAD'):
    """
    Create risk metrics summary table
    
    Parameters:
    -----------
    loans_df : pandas.DataFrame
        DataFrame with risk metrics
    pd_col : str
        Column name for probability of default
    el_col : str
        Column name for expected loss
    rwa_col : str
        Column name for risk-weighted assets
    ead_col : str
        Column name for exposure at default
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with risk metrics summary
    """
    # Calculate risk metrics
    total_exposure = loans_df[ead_col].sum()
    total_rwa = loans_df[rwa_col].sum()
    total_el = loans_df[el_col].sum()
    weighted_avg_pd = (loans_df[pd_col] * loans_df[ead_col]).sum() / total_exposure
    
    # Categorize loans by risk
    loans_df['RiskCategory'] = pd.cut(loans_df[pd_col], 
                                     bins=[0, 0.05, 0.1, 0.3, 1.0],
                                     labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Calculate exposure and expected loss by risk category
    risk_summary = loans_df.groupby('RiskCategory').agg({
        ead_col: 'sum',
        el_col: 'sum'
    }).reset_index()
    
    risk_summary['Exposure %'] = risk_summary[ead_col] / total_exposure * 100
    risk_summary['EL % of Exposure'] = risk_summary[el_col] / risk_summary[ead_col] * 100
    
    # Create metrics summary
    metrics = pd.DataFrame([{
        'Total Exposure': f"${total_exposure:,.2f}",
        'Total RWA': f"${total_rwa:,.2f}",
        'Total EL': f"${total_el:,.2f}",
        'EL % of Exposure': f"{(total_el/total_exposure*100):.2f}%",
        'Weighted Avg PD': f"{weighted_avg_pd:.2f}%"
    }])
    
    return metrics, risk_summary