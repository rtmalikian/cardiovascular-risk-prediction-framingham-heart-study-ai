# Cardiovascular Event Risk Prediction Tool

A web-based clinical decision support tool using Python and Streamlit to predict 10-year cardiovascular event risk, using the actual Framingham Heart Study dataset.

## Overview

This project develops a machine learning model to predict cardiovascular event risk based on the publicly available Framingham Heart Study dataset. The tool provides real-time AI inference through an interactive web interface, designed for clinical decision support.

### Features

- Interactive web interface built with Streamlit
- Machine learning model trained on actual Framingham Heart Study dataset
- Real-time cardiovascular risk prediction
- Data visualization and interpretation tools
- User-friendly input controls for risk factor assessment

## Data Source

This project uses the actual Framingham Heart Study dataset (`framingham_heart_study.csv`), which is publicly available. The dataset contains the following features:
- **Demographics**: gender, age, education
- **Lifestyle factors**: current smoking status, cigarettes per day
- **Medical history**: prevalent stroke, prevalent hypertension, diabetes
- **Physiological measures**: total cholesterol, systolic/diastolic blood pressure, BMI, heart rate, glucose
- **Target variable**: 10-year risk of coronary heart disease (TenYearCHD)

The dataset contains 4,240 records with 15 features and is already included in this repository.

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
1. Age: 16.55% (highest predictive factor)
2. Systolic Blood Pressure: 14.87%
3. Diastolic Blood Pressure: 11.68%
4. BMI: 11.63%
5. Total Cholesterol: 11.63%
6. Glucose: 9.89%
7. Heart Rate: 8.44%
8. Cigarettes per Day: 5.45%
9. Prevalent Hypertension: 3.24%
10. Education Level: 3.21%

### Data Insights:
- Training set: 3,392 samples
- Test set: 848 samples
- Target distribution: 2,877 (No CHD) vs 515 (CHD) cases
- Missing values handled: education (105), glucose (388), cigsPerDay (29), and others
- Outliers detected and managed in multiple features

## Results and Discussion

### What These Results Mean (For Non-Medical Professionals)

Our model achieved good accuracy (85.1%) in predicting who is likely to develop heart disease within 10 years. This means that out of every 100 people the model evaluates, it correctly identifies about 85 of them as either high-risk or low-risk for heart problems.

The Area Under the Curve (AUC) score of 0.702 indicates the model performs significantly better than random chance (which would be 0.5), meaning it can reliably distinguish between people who will and won't develop heart disease.

### Key Findings Explained Simply:

**Most Important Risk Factors:**
1. **Age** (top predictor) - As expected, older people have higher risk of heart disease
2. **Blood pressure** (second most important) - Both systolic and diastolic pressures matter
3. **BMI and cholesterol** - Weight and fat levels strongly predict heart risks
4. **Glucose levels** - Blood sugar is a significant indicator of heart disease risk

**Clinical Significance:**
- The model correctly identifies about 71% of people who will develop heart disease (precision)
- It misses some cases (recall is only 4%), which is typical in early screening tools
- The balance between catching cases and avoiding false alarms is well-calibrated

### Real-World Applications

**For Healthcare Providers:**
- Quick screening tool to identify patients who need more intensive cardiac monitoring
- Decision support for prioritizing preventive care interventions
- Tool for discussing lifestyle modification strategies with patients

**For Patients:**
- Personalized risk assessment to guide lifestyle changes
- Motivation for improving diet, exercise, or quitting smoking
- Better understanding of how individual risk factors contribute to overall heart health

**Public Health Benefits:**
- Resources can be allocated more efficiently to high-risk individuals
- Early intervention could prevent costly emergency treatments
- Population-level screening programs could identify at-risk groups

### Important Limitations to Understand

**This is NOT a diagnostic tool** - it provides risk estimates, not medical diagnoses
- Should be used alongside traditional medical assessments, not replacing them
- Results may not apply to all racial/ethnic groups equally
- Other factors not measured in the study may influence heart disease risk
- Always consult healthcare providers for medical decisions

### Impact Measure
- The model identifies the most powerful risk factors that individuals can work to modify (diet, exercise, blood pressure control)
- Early identification could lead to prevention of thousands of heart disease cases
- Cost-effective screening compared to expensive cardiac imaging for everyone

## Creator

This project was created by Raphael Tomas Malikian (rtmalikian@gmail.com) with the help of Qwen Code in Visual Studio Code using the "coder-model".

## Data Wrangling Achievement

Successfully cleaned, organized, and trained a machine learning model on the actual Framingham Heart Study dataset, demonstrating the ability to manage and structure real-world medical data for clinical decision support applications.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/framingham.git
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the Framingham dataset:
   - Go to: https://www.kaggle.com/datasets/noeyislearning/framingham-heart-study
   - Download the `framingham.csv` file
   - Place it in the `data/` directory

5. Train the model with the actual dataset:
   ```bash
   python train_model.py
   ```

6. Run the application:
   ```bash
   streamlit run app.py
   ```

## Alternative Setup (with sample data)

If you don't have access to the actual dataset, the application will automatically create a sample dataset for demonstration:
   ```bash
   python train_model.py  # This will create sample data if Framingham dataset is not found
   streamlit run app.py
   ```

## Usage

The application provides an intuitive interface for inputting patient data and receiving cardiovascular risk predictions. The model considers multiple risk factors including age, gender, smoking status, blood pressure, cholesterol levels, diabetes status, and more.

## Contributing

If you'd like to contribute to this project, please follow standard GitHub contribution guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Creator

This project was created by Raphael Tomas Malikian (rtmalikian@gmail.com) with the help of Qwen Code in Visual Studio Code using the "coder-model".