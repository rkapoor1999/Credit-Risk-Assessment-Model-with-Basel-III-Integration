import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def select_features(loans_df):
    """
    Select relevant features for modeling
    
    Parameters:
    -----------
    loans_df : pandas.DataFrame
        Loan data
        
    Returns:
    --------
    pandas.DataFrame
        Data with selected features
    """
    features = ['loan_amnt', 'term', 'int_rate', 'installment', 'grade',
              'emp_length', 'home_ownership', 'annual_inc', 'dti',
              'delinq_2yrs', 'fico_range_low', 'fico_range_high', 'inq_last_6mths',
              'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc']
    
    return loans_df[features]

def encode_categorical(feature_df):
    """
    Encode categorical variables using one-hot encoding
    
    Parameters:
    -----------
    feature_df : pandas.DataFrame
        Data with selected features
        
    Returns:
    --------
    pandas.DataFrame
        Data with encoded categorical features
    """
    return pd.get_dummies(feature_df, drop_first=True)

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test):
    """
    Scale numerical features using StandardScaler
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature matrix
    X_test : pandas.DataFrame
        Testing feature matrix
        
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    numerical_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    return X_train_scaled, X_test_scaled, scaler

def handle_missing_values(df, zero_fill_cols=None, mode_fill_cols=None, median_fill_cols=None, 
                          drop_threshold=0.5, group_fill_cols=None):
    """
    Handle missing values in the dataset using various strategies
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe with missing values
    zero_fill_cols : list, optional
        Columns to fill with zeros (typically delinquency or count columns)
    mode_fill_cols : list, optional
        Columns to fill with mode (typically categorical columns)
    median_fill_cols : list, optional
        Columns to fill with median (typically continuous numeric columns)
    drop_threshold : float, optional
        Drop columns with missing ratio greater than this threshold
    group_fill_cols : dict, optional
        Dictionary with format {target_col: grouping_col} for grouped imputation
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with handled missing values
    list
        List of dropped columns
    """
    df_processed = df.copy()
    
    # Drop columns with too many missing values
    if drop_threshold:
        missing_ratio = df_processed.isnull().sum() / len(df_processed)
        cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index.tolist()
        df_processed = df_processed.drop(columns=cols_to_drop)
        print(f"Dropped {len(cols_to_drop)} columns with > {drop_threshold*100}% missing values")
    else:
        cols_to_drop = []
    
    # Fill missing with zeros (common for count features)
    if zero_fill_cols:
        for col in zero_fill_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(0)
    
    # Fill missing with mode (common for categorical features)
    if mode_fill_cols:
        for col in mode_fill_cols:
            if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    # Fill missing with median (common for continuous features)
    if median_fill_cols:
        for col in median_fill_cols:
            if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Group-based imputation (e.g., fill DTI based on grade)
    if group_fill_cols:
        for target_col, group_col in group_fill_cols.items():
            if target_col in df_processed.columns and group_col in df_processed.columns:
                df_processed[target_col] = df_processed.groupby(group_col)[target_col].transform(
                    lambda x: x.fillna(x.median() if x.dtype.kind in 'fc' else x.mode()[0])
                )
    
    return df_processed, cols_to_drop