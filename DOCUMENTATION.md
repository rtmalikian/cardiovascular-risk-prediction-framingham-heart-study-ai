# Cardiovascular Event Risk Prediction Tool

## Project Overview

This project develops a web-based clinical decision support tool using Python and Streamlit to predict 10-year cardiovascular event risk, aligning model output with the Framingham Heart Study large dataset.

### Key Features
- Interactive web interface built with Streamlit
- Machine learning model trained on Framingham Heart Study dataset
- Real-time cardiovascular risk prediction
- Data visualization and interpretation tools
- User-friendly input controls for risk factor assessment

## Data Wrangling Achievement

Successfully cleaned, organized, and trained a machine learning model on a large, complex medical database, demonstrating the ability to manage and structure highly heterogeneous medical data for real-time AI inference.

## Project Structure

```
framingham/
├── app.py                    # Main Streamlit application
├── train_model.py           # Model training script
├── requirements.txt         # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # License information
├── src/                    # Source code modules
│   ├── data_processing.py  # Data preprocessing functions
│   └── ml_model.py         # Machine learning model class
├── data/                   # Data files
│   └── framingham.csv      # Framingham dataset
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