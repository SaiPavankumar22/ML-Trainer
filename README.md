# ML-Train: Automated Machine Learning Platform

ML-Train is a comprehensive web-based platform that automates the entire machine learning pipeline, from data preprocessing to model deployment. It's designed to make machine learning accessible to users of all skill levels.

## 🌟 Key Features

### 1. Data Preprocessing
- **Automated Data Cleaning**
  - Missing value handling
  - Outlier detection and treatment
  - Duplicate removal
  - Data type conversion
- **Feature Engineering**
  - Categorical encoding (One-Hot, Label)
  - Feature scaling (Standard, Min-Max)
  - Polynomial feature generation
- **Data Validation**
  - Format verification
  - Data type checking
  - Value range validation

### 2. Model Training
- **Automated Model Selection**
  - Task type detection (Classification/Regression)
  - Best model recommendation
  - Hyperparameter optimization
- **Multiple Model Support**
  - Regression Models:
    - Linear Regression
    - Polynomial Regression
    - Ridge Regression
    - Lasso Regression
    - Elastic Net
    - Support Vector Regression (SVR)
    - Decision Tree Regressor
    - Random Forest Regressor
    - XGBoost Regressor
    - Gradient Boosting Regressor
    - AdaBoost Regressor
    - CatBoost Regressor
    - LightGBM Regressor

  - **Classification Models**:
    - Logistic Regression
    - Ridge Classifier
    - Support Vector Classifier (SVC)
    - Decision Tree Classifier
    - Random Forest Classifier
    - Gradient Boosting Classifier
    - AdaBoost Classifier
    - XGBoost Classifier
    - LightGBM Classifier
    - CatBoost Classifier
    - Naive Bayes variants
    - K-Nearest Neighbors

### 3. Model Export
- **Multiple Format Support**
  - Pickle (.pkl)
  - Joblib (.joblib)
  - ONNX (.onnx)
  - PMML (.pmml)
  - CoreML (.mlmodel)
  - TensorFlow (.pb)
  - PyTorch (.pt)
- **Format-specific Optimizations**
  - ONNX: Cross-platform compatibility
  - PMML: Enterprise integration
  - CoreML: iOS deployment
  - TensorFlow/PyTorch: Deep learning support

### 4. Real-time Predictions
- **Interactive Prediction Interface**
  - Dynamic form generation based on model features
  - Real-time prediction results
  - Support for all model formats
- **Prediction Features**
  - Single instance predictions
  - Batch predictions
  - Probability scores for classification
  - Confidence intervals
- **User-friendly Input**
  - Input validation
  - Error handling
  - Clear result presentation

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ml-train.git
   cd ml-train
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## 📊 Usage Guide

### 1. Data Upload
1. Click "Upload Dataset" on the homepage
2. Select your CSV or Excel file
3. Choose the target column
4. Click "Upload"

### 2. Data Preprocessing
1. Review the automated preprocessing steps
2. Adjust parameters if needed
3. Click "Process Data"

### 3. Model Training
1. Select model type (Auto/Manual)
2. Choose specific model (if manual)
3. Set hyperparameters
4. Click "Train Model"

### 4. Model Export
1. Select desired export format
2. Click "Export Model"
3. Download the model file

### 5. Making Predictions
1. Navigate to the predictions page
2. Select your trained model
3. Enter input values for each feature
4. Click "Predict" to get results
5. View prediction results and probabilities (if available)

## 🔧 Technical Details

### Backend Architecture
- Flask web framework
- RESTful API design
- Modular code structure
- Error handling and logging

### Data Processing Pipeline
1. Data validation
2. Type conversion
3. Missing value handling
4. Feature engineering
5. Model training
6. Evaluation
7. Export

### Model Export Process
1. Model serialization
2. Format conversion
3. Optimization
4. Validation
5. Export

### Prediction System
1. Model loading and validation
2. Input preprocessing
3. Prediction generation
4. Result formatting
5. Error handling

## 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📧 Contact
For questions and support, please open an issue in the GitHub repository. 