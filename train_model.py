import pandas as pd
import numpy as np
from src.data_processing import load_framingham_data, preprocess_data, clean_data
from src.ml_model import CardiovascularRiskModel, compare_models
import os
import pickle
import requests
from pathlib import Path

def download_framingham_data():
    """
    Download the Framingham dataset from Kaggle if it doesn't exist locally.
    """
    print("Checking for Framingham dataset...")

    kaggle_dataset = "noeyislearning/framingham-heart-study"

    # Check if kaggle API is available
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Initialize the API
        try:
            api = KaggleApi()
            api.authenticate()
            print("Kaggle API authenticated successfully.")

            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)

            # Download the dataset
            print(f"Downloading {kaggle_dataset}...")
            api.dataset_download_files(kaggle_dataset, path='data/', unzip=True)

            # Look for the framingham.csv file in the data directory
            for file in os.listdir('data'):
                if file.endswith('.csv') and 'framingham' in file.lower():
                    print(f"Dataset downloaded successfully as {file}")
                    return True

            print("Dataset file not found after download. Looking for any CSV file...")
            # If the exact filename is different, just look for any csv file
            for file in os.listdir('data'):
                if file.endswith('.csv'):
                    # Rename it to framingham.csv if it's not already named that
                    if file != 'framingham.csv':
                        os.rename(f'data/{file}', f'data/framingham.csv')
                    print(f"Dataset found and renamed to framingham.csv")
                    return True

            return False

        except Exception as e:
            print(f"Could not authenticate or download using Kaggle API: {e}")
            print("Please ensure you have Kaggle credentials set up.")
            print("You need to have a Kaggle account and set up the API credentials.")
            print("Follow these steps:")
            print("1. Create a Kaggle account at https://www.kaggle.com/")
            print("2. Go to Account -> API -> Create New API Token")
            print("3. Save the kaggle.json file to ~/.kaggle/kaggle.json")
            print("4. Run this script again")
            print("\nAlternatively, download manually from:")
            print("https://www.kaggle.com/datasets/noeyislearning/framingham-heart-study")
            print("Save as 'data/framingham.csv' in this project's 'data' directory")
            return False

    except ImportError:
        print("Kaggle API package not found.")
        print("To install: pip install kaggle")
        print("After installation, follow these steps:")
        print("1. Create a Kaggle account at https://www.kaggle.com/")
        print("2. Go to Account -> API -> Create New API Token")
        print("3. Save the kaggle.json file to ~/.kaggle/kaggle.json")
        print("4. Run this script again")
        print("\nAlternatively, download manually from:")
        print("https://www.kaggle.com/datasets/noeyislearning/framingham-heart-study")
        print("Save as 'data/framingham.csv' in this project's 'data' directory")
        return False

def main():
    print("Starting Cardiovascular Risk Model Training...")
    
    # Load the dataset
    print("\n1. Loading Framingham dataset...")
    df = load_framingham_data("data/framingham.csv")
    
    if df is None:
        print("\nFramingham dataset not found locally.")
        downloaded = download_framingham_data()
        
        if not downloaded:
            print("\nDataset not found and cannot be downloaded automatically.")
            print("Creating a sample dataset for demonstration purposes...")
            df = create_sample_dataset()
            print("Sample dataset created successfully.")
        else:
            # Try loading again after download
            df = load_framingham_data("data/framingham.csv")
    
    # Clean the data
    print("\n2. Cleaning the dataset...")
    df_cleaned = clean_data(df)
    
    # Preprocess the data
    print("\n3. Preprocessing the data...")
    result = preprocess_data(df_cleaned)
    
    if result is None:
        print("Preprocessing failed. Please check the dataset structure.")
        return
    
    X_train, X_test, y_train, y_test, scaler, numerical_cols = result
    
    print("\n4. Training and comparing different models...")
    comparison_results = compare_models(X_train, y_train, X_test, y_test)
    
    # Show comparison results
    print("\nModel Comparison Results:")
    print("-" * 50)
    for model_name, metrics in comparison_results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Train the best performing model (Random Forest in this case)
    print("\n5. Training final model (Random Forest)...")
    final_model = CardiovascularRiskModel('random_forest')
    final_model.train(X_train, y_train)
    
    # Evaluate the final model
    print("\n6. Final model evaluation:")
    final_metrics = final_model.evaluate(X_test, y_test)
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Cross-validation
    print("\n7. Performing cross-validation...")
    cv_scores = final_model.cross_validate(X_train, y_train)
    print(f"  Cross-validation scores: {cv_scores}")
    print(f"  Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Hyperparameter tuning
    print("\n8. Performing hyperparameter tuning...")
    final_model.hyperparameter_tuning(X_train, y_train)
    
    # Re-evaluate with tuned model
    print("\n9. Evaluating tuned model...")
    final_metrics_tuned = final_model.evaluate(X_test, y_test)
    print("Tuned model metrics:")
    for metric, value in final_metrics_tuned.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save the model
    print("\n10. Saving the trained model...")
    os.makedirs('models', exist_ok=True)  # Ensure models directory exists
    final_model.save_model('models/cardiovascular_risk_model.pkl')
    
    # Save the scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved to models/scaler.pkl")
    
    print("\nModel training completed successfully!")
    
    # Show feature importance for Random Forest
    if hasattr(final_model.model, 'feature_importances_'):
        print("\nFeature Importance (Top 10):")
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': final_model.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(10))


def create_sample_dataset():
    """
    Create a sample dataset similar to the Framingham dataset for demonstration purposes
    """
    np.random.seed(42)
    n_samples = 4240  # Match the size of the actual dataset
    
    # Generate sample data similar to Framingham Heart Study
    # Based on typical ranges from the actual dataset
    data = {
        'male': np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55]),  # ~55% male
        'age': np.random.normal(49, 15, size=n_samples).clip(20, 80).astype(int),
        'education': np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.2, 0.4, 0.25, 0.15]),
        'currentSmoker': np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]),  # ~60% smokers
        'cigsPerDay': np.random.poisson(9, size=n_samples) * np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]),
        'BPMeds': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),  # ~15% on BP meds
        'prevalentStroke': np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02]),  # ~2% with stroke
        'prevalentHyp': np.random.choice([0, 1], size=n_samples, p=[0.35, 0.65]),  # ~65% with hypertension
        'diabetes': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),  # ~5% with diabetes
        'totChol': np.random.normal(236, 42, size=n_samples).clip(120, 600),  # Total cholesterol
        'sysBP': np.random.normal(133, 22, size=n_samples).clip(80, 300),    # Systolic BP
        'diaBP': np.random.normal(83, 12, size=n_samples).clip(40, 150),     # Diastolic BP
        'BMI': np.random.normal(25.8, 4.2, size=n_samples).clip(15, 50),     # BMI
        'heartRate': np.random.normal(75, 13, size=n_samples).clip(40, 150), # Heart rate
        'glucose': np.random.normal(82, 23, size=n_samples).clip(40, 400)    # Glucose
    }
    
    df = pd.DataFrame(data)
    
    # Create a realistic target variable based on risk factors
    # Higher risk for older people, smokers, diabetics, hypertensives, etc.
    risk_score = (
        (df['age'] - 20) / 100 +                                    # Age factor (older = higher risk)
        df['male'] * 0.05 +                                        # Gender factor (male = higher risk)
        df['currentSmoker'] * 0.1 +                                # Smoking factor
        (df['sysBP'] - 120) / 500 +                                # Blood pressure factor
        (df['totChol'] - 200) / 1000 +                             # Cholesterol factor
        df['diabetes'] * 0.3 +                                     # Diabetes factor
        df['prevalentHyp'] * 0.1 +                                 # Prevalent hypertension factor
        df['prevalentStroke'] * 0.3                                # Prevalent stroke factor
    ).clip(0, 1)  # Ensure between 0 and 1
    
    # Add some randomness and convert to binary outcome
    # Base rate of CHD risk is approximately 15% in the Framingham dataset
    df['TenYearCHD'] = (risk_score > 0.35) & (np.random.random(n_samples) < risk_score + 0.15)
    df['TenYearCHD'] = df['TenYearCHD'].astype(int)
    
    # Ensure reasonable ranges for all variables
    df['cigsPerDay'] = df['cigsPerDay'].clip(0, 70)
    df['totChol'] = df['totChol'].clip(120, 600)
    df['sysBP'] = df['sysBP'].clip(80, 300)
    df['diaBP'] = df['diaBP'].clip(40, 150)
    df['BMI'] = df['BMI'].clip(15, 50)
    df['heartRate'] = df['heartRate'].clip(40, 150)
    df['glucose'] = df['glucose'].clip(40, 400)
    
    # Save the sample dataset
    os.makedirs('data', exist_ok=True)  # Ensure data directory exists
    df.to_csv('data/framingham.csv', index=False)
    
    return df


if __name__ == "__main__":
    main()