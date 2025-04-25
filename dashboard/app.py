import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.basel.stress_testing import apply_stress_scenario
from src.basel.risk_weights import assign_basel_risk_weight
from src.visualization.dashboard import (
    plot_pd_distribution, 
    compare_pd_distributions, 
    plot_risk_weight_distribution,
    plot_expected_loss_by_risk,
    plot_capital_requirements,
    create_risk_metrics_table
)

# Load preprocessed data
risk_data = pd.read_csv('../data/processed/basel_risk_calculations.csv')

# Initialize app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Credit Risk Assessment Dashboard with Basel III Integration"),
    
    html.Div([
        html.H3("Stress Testing Scenarios"),
        dcc.RadioItems(
            id='scenario-selection',
            options=[
                {'label': 'Normal', 'value': 'normal'},
                {'label': 'Mild Stress', 'value': 'mild'},
                {'label': 'Moderate Stress', 'value': 'moderate'},
                {'label': 'Severe Stress', 'value': 'severe'},
            ],
            value='normal'
        ),
    ]),
    
    html.Div([
        html.Div([
            html.H3("Risk Metrics"),
            html.Div(id='risk-metrics-output')
        ], className='six columns'),
        
        html.Div([
            html.H3("Capital Requirements"),
            html.Div(id='capital-requirements-output')
        ], className='six columns'),
    ], className='row'),
    
    html.Div([
        html.H3("PD Distribution"),
        dcc.Graph(id='pd-histogram')
    ]),
    
    html.Div([
        html.H3("Expected Loss by Risk Category"),
        dcc.Graph(id='el-by-risk')
    ]),
    
    html.Div([
        html.H3("Risk Weight Distribution"),
        dcc.Graph(id='risk-weight-distribution')
    ]),
    
    html.Div([
        html.H3("Basel III Overview"),
        html.Div([
            html.H4("Key Components"),
            html.Ul([
                html.Li("Probability of Default (PD): The likelihood that a borrower will default"),
                html.Li("Loss Given Default (LGD): The estimated loss if a default occurs"),
                html.Li("Exposure at Default (EAD): The expected exposure when default occurs"),
                html.Li("Risk-Weighted Assets (RWA): Assets weighted by their risk level"),
                html.Li("Expected Loss (EL): PD × LGD × EAD"),
            ]),
            html.H4("Capital Requirements"),
            html.Ul([
                html.Li("Tier 1 Capital: 6% of RWA"),
                html.Li("Total Capital: 8% of RWA"),
                html.Li("Conservation Buffer: Additional 2.5% of RWA"),
            ]),
        ])
    ])
])

# Define callbacks
@app.callback(
    [Output('risk-metrics-output', 'children'),
     Output('capital-requirements-output', 'children'),
     Output('pd-histogram', 'figure'),
     Output('el-by-risk', 'figure'),
     Output('risk-weight-distribution', 'figure')],
    [Input('scenario-selection', 'value')]
)
def update_dashboard(scenario):
    # Apply selected stress scenario
    if scenario == 'normal':
        current_pd = risk_data['PD']
        current_el = risk_data['EL']
        current_rwa = risk_data['RWA']
        current_risk_weight = risk_data['RiskWeight']
    else:
        current_pd = apply_stress_scenario(risk_data['PD'], scenario=scenario)
        current_el = current_pd * risk_data['LGD'] * risk_data['EAD']
        current_risk_weight = current_pd.apply(assign_basel_risk_weight)
        current_rwa = risk_data['EAD'] * current_risk_weight
    
    # Calculate risk metrics
    total_exposure = risk_data['EAD'].sum()
    total_rwa = current_rwa.sum()
    total_el = current_el.sum()
    
    # Create risk metrics table
    temp_df = risk_data.copy()
    temp_df['PD'] = current_pd
    temp_df['EL'] = current_el
    temp_df['RWA'] = current_rwa
    temp_df['RiskWeight'] = current_risk_weight
    
    metrics_df, risk_summary = create_risk_metrics_table(temp_df)
    
    # Format risk metrics table
    risk_metrics = html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in metrics_df.columns])
        ),
        html.Tbody([
            html.Tr([html.Td(metrics_df.iloc[0][col]) for col in metrics_df.columns])
        ])
    ])
    
    # Calculate capital requirements
    tier1_capital_ratio = 0.06  # 6%
    total_capital_ratio = 0.08  # 8%
    conservation_buffer = 0.025  # 2.5%
    
    min_tier1_capital = total_rwa * tier1_capital_ratio
    min_total_capital = total_rwa * total_capital_ratio
    min_capital_with_buffer = total_rwa * (total_capital_ratio + conservation_buffer)
    
    # Format capital requirements table
    capital_requirements = html.Table([
        html.Thead(
            html.Tr([html.Th("Capital Component"), html.Th("Amount"), html.Th("% of RWA")])
        ),
        html.Tbody([
            html.Tr([html.Td("Tier 1 Capital"), html.Td(f"${min_tier1_capital:,.2f}"), html.Td(f"{tier1_capital_ratio*100:.1f}%")]),
            html.Tr([html.Td("Total Capital"), html.Td(f"${min_total_capital:,.2f}"), html.Td(f"{total_capital_ratio*100:.1f}%")]),
            html.Tr([html.Td("With Conservation Buffer"), html.Td(f"${min_capital_with_buffer:,.2f}"), html.Td(f"{(total_capital_ratio+conservation_buffer)*100:.1f}%")])
        ])
    ])
    
    # Create PD histogram
    pd_hist = plot_pd_distribution(current_pd, title=f'PD Distribution - {scenario.capitalize()} Scenario')
    
    # Create Expected Loss by Risk Category chart
    el_risk_fig = plot_expected_loss_by_risk(current_pd, current_el, title=f'Expected Loss by Risk Category - {scenario.capitalize()} Scenario')
    
    # Create Risk Weight Distribution chart
    risk_weight_fig = plot_risk_weight_distribution(current_risk_weight, risk_data['EAD'], title=f'Distribution of Exposure by Risk Weight - {scenario.capitalize()} Scenario')
    
    return risk_metrics, capital_requirements, pd_hist, el_risk_fig, risk_weight_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)