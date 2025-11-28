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

This project uses the actual Framingham Heart Study dataset, which is publicly available. The dataset contains the following features:
- **Demographics**: gender, age, education
- **Lifestyle factors**: current smoking status, cigarettes per day
- **Medical history**: prevalent stroke, prevalent hypertension, diabetes
- **Physiological measures**: total cholesterol, systolic/diastolic blood pressure, BMI, heart rate, glucose
- **Target variable**: 10-year risk of coronary heart disease (TenYearCHD)

To use the actual dataset:
1. Download from [Kaggle](https://www.kaggle.com/datasets/noeyislearning/framingham-heart-study)
2. Place the `framingham.csv` file in the `data/` directory
3. Run the training script to build the model with real data

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