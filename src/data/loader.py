import pandas as pd

def load_loan_data(file_path):
    loans_df = pd.read_csv(file_path)
    print(f"Loaded {loans_df.shape[0]} loans with {loans_df.shape[1]} features")
    return loans_df

def create_target_variable(loans_df):
    return loans_df['loan_status'].apply(
        lambda x: 1 if x in ['Charged Off', 'Default', 'Late (31-120 days)'] else 0)