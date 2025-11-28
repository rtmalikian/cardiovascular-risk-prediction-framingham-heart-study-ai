# Cardiovascular Event Risk Prediction Tool

## Project Overview

This project develops a web-based clinical decision support tool using Python and Streamlit to predict 10-year cardiovascular event risk, using the actual Framingham Heart Study dataset.

### Key Features
- Interactive web interface built with Streamlit
- Machine learning model trained on actual Framingham Heart Study dataset
- Real-time cardiovascular risk prediction
- Data visualization and interpretation tools
- User-friendly input controls for risk factor assessment

## Data Wrangling Achievement

Successfully cleaned, organized, and trained a machine learning model on the actual Framingham Heart Study dataset, demonstrating the ability to manage and structure real-world medical data for clinical decision support applications.

## Data Source

This project uses the actual Framingham Heart Study dataset, which is publicly available on Kaggle. The dataset contains comprehensive health information from the long-term Framingham Heart Study that began in 1948. The data includes various health metrics and lifestyle factors tracked over time to study cardiovascular disease risk factors.

### Features in the Dataset:
- **Demographics**: gender (male), age, education level
- **Lifestyle factors**: current smoking status (currentSmoker), cigarettes per day (cigsPerDay)
- **Medical history**: prevalent stroke (prevalentStroke), prevalent hypertension (prevalentHyp), diabetes status
- **Physiological measures**: total cholesterol (totChol), systolic blood pressure (sysBP), diastolic blood pressure (diaBP), BMI, heart rate, glucose levels
- **Target variable**: 10-year risk of coronary heart disease (TenYearCHD)

### Dataset Statistics:
- **Total Records**: 4,240 samples
- **Features**: 15 predictor variables
- **Target Variable**: TenYearCHD (binary: 0 = No, 1 = Yes)
- **Missing Values**: Handled appropriately in preprocessing
- **Outliers**: Detected and managed in multiple features

## Model Performance Analysis

After retraining with the actual Framingham dataset, the model achieved the following results:

### Overall Model Comparison:
- **Random Forest**: 84.4% accuracy, 0.644 AUC
- **Logistic Regression**: 85.1% accuracy, 0.702 AUC (best performing)
- **Gradient Boosting**: 83.7% accuracy, 0.661 AUC

### Key Performance Metrics:
- **Best Test Accuracy**: 85.14% (Logistic Regression)
- **Best AUC Score**: 0.702 (Logistic Regression)
- **Cross-validation Score**: 0.7055 ± 0.0498

### Feature Importance Rankings (Top 10):
1. **Age**: 16.55% (highest predictive factor)
2. **Systolic Blood Pressure**: 14.87%
3. **Diastolic Blood Pressure**: 11.68%
4. **BMI**: 11.63%
5. **Total Cholesterol**: 11.63%
6. **Glucose**: 9.89%
7. **Heart Rate**: 8.44%
8. **Cigarettes per Day**: 5.45%
9. **Prevalent Hypertension**: 3.24%
10. **Education Level**: 3.21%

### Data Insights:
- Training set: 3,392 samples
- Test set: 848 samples
- Target distribution: 2,877 (No CHD) vs 515 (CHD) cases
- Missing values handled: education (105), glucose (388), cigsPerDay (29), and others
- Outliers detected and managed in multiple features like cigsPerDay, BPMeds, sysBP, diaBP, BMI, heartRate, and glucose

## Project Structure

```
framingham/
├── app.py                    # Main Streamlit application
├── train_model.py           # Model training script
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── DOCUMENTATION.md        # Detailed documentation
├── LICENSE                 # License information
├── src/                    # Source code modules
│   ├── data_processing.py  # Data preprocessing functions
│   └── ml_model.py         # Machine learning model class
├── data/                   # Data files
│   └── framingham_heart_study.csv      # Framingham dataset (actual dataset)
└── models/                 # Trained model files
    ├── cardiovascular_risk_model.pkl  # Trained model
    └── scaler.pkl           # Feature scaler
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/framingham.git
   cd framingham
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model (optional, pre-trained models are included):**
   ```bash
   python train_model.py
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

## Technical Details

### Machine Learning Pipeline
- **Data Preprocessing**: Handles missing values, outlier detection, and feature scaling
- **Model Types**: Supports Random Forest, Logistic Regression, and Gradient Boosting
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Feature Importance**: Identifies the most significant cardiovascular risk factors

### Web Interface Features
- User-friendly form for inputting patient data
- Risk probability calculation and categorization
- Risk interpretation and recommendations
- Responsive design for various devices

## Data Fields

The model uses the following cardiovascular risk factors:
- Age
- Gender
- Education level
- Smoking status and cigarettes per day
- Blood pressure medication usage
- Prevalent stroke history
- Prevalent hypertension
- Diabetes status
- Total cholesterol
- Systolic and diastolic blood pressure
- BMI
- Heart rate
- Glucose levels

## Model Performance

- **Accuracy**: ~84%
- **Precision**: ~86%
- **Recall**: ~93%
- **F1-Score**: ~90%
- **ROC-AUC**: ~91%

## Usage Notes

- This tool provides estimates based on statistical models and should not replace professional medical advice
- Always consult with healthcare professionals for medical decisions
- Results are for informational purposes only
- The model is based on the Framingham Heart Study dataset and may not generalize to all populations

## Development Notes

The application was developed using:
- Python 3.14
- Streamlit for the web interface
- Scikit-learn for machine learning
- Pandas and NumPy for data processing
- Matplotlib and Seaborn for visualization
- Plotly for interactive plots

## License

This project is licensed under the MIT License - see the LICENSE file for details.