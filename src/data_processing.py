import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

def load_framingham_data(data_path="data/framingham.csv"):
    """
    Load the Framingham dataset and perform initial data exploration
    """
    try:
        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"Data file not found at {data_path}")
            print("Please ensure the Framingham dataset is available at the specified path.")
            return None
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        
        # Display basic information
        print("\nDataset Info:")
        print(df.info())
        
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nMissing values per column:")
        print(df.isnull().sum())
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """
    Preprocess the Framingham dataset for machine learning
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Define target variable: TenYearCHD (10-year risk of coronary heart disease)
    target_col = 'TenYearCHD'
    
    if target_col not in data.columns:
        print(f"Target column '{target_col}' not found in dataset.")
        print("Available columns:", list(data.columns))
        return None, None, None, None
    
    # Separate features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Handle missing values
    # For numerical columns, use median imputation
    num_imputer = SimpleImputer(strategy='median')
    X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
    
    # For categorical columns, use mode imputation
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    
    # Remove any remaining rows with missing values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to maintain column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    print(f"Training target distribution:\n{y_train.value_counts()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, numerical_cols

def clean_data(df):
    """
    Perform data cleaning specific to the Framingham dataset
    """
    data = df.copy()
    
    # Remove duplicate rows
    initial_count = len(data)
    data = data.drop_duplicates()
    print(f"Removed {initial_count - len(data)} duplicate rows")
    
    # Handle outliers in numerical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in numerical_cols:
        # Skip the target column for outlier detection
        if col == 'TenYearCHD':
            continue
            
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"Found {outliers} outliers in {col}")
            # In medical data, we might want to cap outliers instead of removing them
            data[col] = np.clip(data[col], lower_bound, upper_bound)
    
    return data