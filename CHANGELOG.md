# Changelog

All notable changes to the Cardiovascular Risk Prediction Tool project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2025-01-28
### Added
- Integrated actual Framingham Heart Study dataset (4,240 records)
- Added application screenshots (home.png and results.png) to visual documentation
- Implemented hyperparameter tuning for improved model performance
- Added comprehensive feature importance analysis
- Included clinical interpretation of model results

### Changed
- Updated all data processing functions to work with real Framingham dataset
- Enhanced model training with actual clinical data (was using sample data before)
- Improved README with actual performance metrics and clinical validation results
- Updated model performance: Logistic Regression now 85.1% accuracy, 0.702 AUC
- Refactored documentation with healthcare professional terminology
- Changed dataset filename reference to 'framingham_heart_study.csv'

### Fixed
- Removed dummy dataset file and replaced with actual Framingham data
- Corrected model evaluation metrics with actual clinical performance
- Fixed data preprocessing pipeline for real clinical variables
- Resolved inconsistencies between sample and actual data structures

### Removed
- Old dummy dataset file (framingham.csv)
- Sample data generation functions (replaced with real data processing)

## [1.0.0] - 2025-01-28
### Added
- Initial release of Cardiovascular Risk Prediction Tool
- Machine learning models (Random Forest, Logistic Regression, Gradient Boosting)
- Streamlit web application for clinical risk assessment
- Data processing pipeline for Framingham Heart Study dataset
- Model training and evaluation scripts
- Comprehensive documentation with setup instructions

### Features
- Real-time cardiovascular risk prediction based on 15 clinical risk factors
- Clinical decision support interface for healthcare providers
- Data visualization and risk interpretation tools
- Cross-validation and hyperparameter tuning capabilities
- Feature importance analysis for clinical understanding