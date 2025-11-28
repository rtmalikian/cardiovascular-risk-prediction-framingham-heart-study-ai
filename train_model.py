import pandas as pd
import numpy as np
from src.data_processing import load_framingham_data, preprocess_data, clean_data
from src.ml_model import CardiovascularRiskModel, compare_models
import os
import pickle

def main():
    print("Starting Cardiovascular Risk Model Training...")
    
    # Load the dataset
    print("\n1. Loading Framingham dataset...")
    df = load_framingham_data("data/framingham.csv")
    
    if df is None:
        print("\nDataset not found. Creating a sample dataset for demonstration...")
        df = create_sample_dataset()
        print("Sample dataset created successfully.")
    
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
    n_samples = 1000
    
    # Generate sample data similar to Framingham Heart Study
    data = {
        'male': np.random.choice([0, 1], size=n_samples),
        'age': np.random.randint(30, 80, size=n_samples),
        'education': np.random.choice([1, 2, 3, 4], size=n_samples),
        'currentSmoker': np.random.choice([0, 1], size=n_samples),
        'cigsPerDay': np.random.poisson(5, size=n_samples) * np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]),
        'BPMeds': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
        'prevalentStroke': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]),
        'prevalentHyp': np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
        'diabetes': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
        'totChol': np.random.normal(240, 50, size=n_samples),
        'sysBP': np.random.normal(130, 25, size=n_samples),
        'diaBP': np.random.normal(85, 15, size=n_samples),
        'BMI': np.random.normal(26, 4, size=n_samples),
        'heartRate': np.random.normal(75, 10, size=n_samples),
        'glucose': np.random.normal(100, 25, size=n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create a realistic target variable based on risk factors
    risk_score = (
        (df['age'] - 30) / 50 +  # Age factor
        df['male'] * 0.1 +  # Gender factor
        df['currentSmoker'] * 0.2 +  # Smoking factor
        (df['sysBP'] - 120) / 200 +  # Blood pressure factor
        (df['totChol'] - 200) / 500 +  # Cholesterol factor
        df['diabetes'] * 0.15  # Diabetes factor
    ).clip(0, 1)  # Ensure between 0 and 1
    
    # Add some randomness and convert to binary outcome
    df['TenYearCHD'] = (risk_score > 0.3) & (np.random.random(n_samples) < risk_score + 0.1)
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