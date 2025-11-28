import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle
import os

class CardiovascularRiskModel:
    """
    A class to handle the cardiovascular risk prediction model
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the model
        :param model_type: Type of model to use ('random_forest', 'logistic_regression', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        # Initialize the model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42, n_estimators=100)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(random_state=42, n_estimators=100)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def train(self, X_train, y_train):
        """
        Train the model
        :param X_train: Training features
        :param y_train: Training targets
        """
        print(f"Training {self.model_type} model...")
        
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print(f"{self.model_type} model training completed.")
    
    def predict(self, X):
        """
        Make predictions
        :param X: Features to predict on
        :return: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to values if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        :param X: Features to predict on
        :return: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert to values if it's a DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return self.model.predict_proba(X)[:, 1]  # Return probability of positive class
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        :param X_test: Test features
        :param y_test: Test targets
        :return: Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Convert to values if it's a DataFrame
        X_test_values = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        
        # Make predictions
        y_pred = self.model.predict(X_test_values)
        y_pred_proba = self.model.predict_proba(X_test_values)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        :param X: Features
        :param y: Targets
        :param cv: Number of cross-validation folds
        :return: Cross-validation scores
        """
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        scores = cross_val_score(self.model, X_values, y, cv=cv, scoring='roc_auc')
        return scores
    
    def hyperparameter_tuning(self, X_train, y_train, param_grid=None):
        """
        Perform hyperparameter tuning using GridSearchCV
        :param X_train: Training features
        :param y_train: Training targets
        :param param_grid: Parameter grid for tuning
        """
        if param_grid is None:
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            elif self.model_type == 'logistic_regression':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            elif self.model_type == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
        
        X_train_values = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_values, y_train)
        
        # Update the model with best parameters
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_}")
    
    def save_model(self, filepath):
        """
        Save the trained model to a file
        :param filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model from a file
        :param filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")


def compare_models(X_train, y_train, X_test, y_test):
    """
    Compare different model types
    :param X_train: Training features
    :param y_train: Training targets
    :param X_test: Test features
    :param y_test: Test targets
    :return: Dictionary of model performances
    """
    models = {
        'Random Forest': CardiovascularRiskModel('random_forest'),
        'Logistic Regression': CardiovascularRiskModel('logistic_regression'),
        'Gradient Boosting': CardiovascularRiskModel('gradient_boosting')
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.train(X_train, y_train)
        
        # Evaluate the model
        metrics = model.evaluate(X_test, y_test)
        results[name] = metrics
        
        print(f"{name} Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    return results