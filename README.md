# ML Model Training and Deployment Platform

A comprehensive web-based platform for training, evaluating, and deploying machine learning models with support for multiple model formats and export options.

## Features

### Data Processing
- Upload and process CSV/Excel datasets
- Automatic data preprocessing including:
  - Missing value handling
  - Outlier detection and removal
  - Categorical feature encoding
  - Feature scaling
  - Duplicate removal

### Model Training
- Support for multiple model types:
  - **Regression Models**:
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

### Model Evaluation
- Comprehensive evaluation metrics:
  - **Regression**:
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R² Score
  - **Classification**:
    - Accuracy
    - Precision
    - Recall
    - F1 Score

### Model Export
- Multiple export formats supported:
  - Pickle (.pkl)
  - Joblib (.joblib)
  - Dill (.dill)
  - ONNX (.onnx)
  - CoreML (.mlmodel)

### Advanced Features
- Automatic model selection
- Hyperparameter tuning
- Cross-validation
- Model comparison
- API documentation with Swagger UI
- Secure file handling
- Session management

## Technical Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
```
Flask==3.0.2
flask-swagger-ui==4.11.1
pandas==2.2.1
numpy==1.26.4
scikit-learn==1.4.1.post1
xgboost==2.0.3
catboost==1.2.5
lightgbm==4.3.0
onnx==1.15.0
onnxmltools==1.11.2
skl2onnx==1.16.0
nyoka==5.4.0
joblib==1.3.2
dill==0.3.8
tensorflow==2.15.0
torch==2.2.1
coremltools==7.1
```

### System Requirements
- Minimum 4GB RAM
- 2GB free disk space
- Modern web browser (Chrome, Firefox, Safari, or Edge)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Detailed API Documentation

### 1. File Upload and Preprocessing

#### Upload Dataset
```http
POST /upload
Content-Type: multipart/form-data

file: <dataset_file>
```

**Response:**
```json
{
    "status": "success",
    "filename": "processed_dataset.csv",
    "columns": ["feature1", "feature2", "target"],
    "preprocessing_summary": {
        "missing_values_handled": true,
        "outliers_removed": true,
        "categorical_encoded": true,
        "scaling_applied": true
    }
}
```

### 2. Model Training

#### Train Model
```http
POST /train
Content-Type: application/json

{
    "filename": "processed_dataset.csv",
    "model_type": "regression",  // or "classification"
    "selected_model": "xgboost", // or any other model name
    "test_size": 0.2,
    "hyperparameters": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3
    }
}
```

**Response:**
```json
{
    "selected_model": "XGBoostRegressor",
    "hyperparameters": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3
    },
    "evaluation_metrics": {
        "MSE": 0.123,
        "RMSE": 0.351,
        "R2 Score": 0.876
    },
    "model_files": {
        "pkl": "<base64_encoded_model>",
        "joblib": "<base64_encoded_model>",
        "onnx": "<base64_encoded_model>"
    }
}
```

### 3. Model Download

#### Download Model
```http
GET /api/download/{filename}
```

**Response:**
- File download with appropriate content-type

## Detailed Model Usage Examples

### 1. Regression Example: House Price Prediction

```python
# Example dataset structure
import pandas as pd
data = {
    'area': [1500, 2000, 1200, 1800],
    'bedrooms': [3, 4, 2, 3],
    'bathrooms': [2, 3, 1, 2],
    'price': [300000, 400000, 250000, 350000]
}
df = pd.DataFrame(data)

# Training parameters
params = {
    "model_type": "regression",
    "selected_model": "xgboost",
    "hyperparameters": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3
    }
}

# Expected evaluation metrics
{
    "MSE": 0.123,
    "RMSE": 0.351,
    "R2 Score": 0.876
}
```

### 2. Classification Example: Customer Churn Prediction

```python
# Example dataset structure
import pandas as pd
data = {
    'tenure': [12, 24, 6, 36],
    'monthly_charges': [70.5, 89.9, 29.9, 99.9],
    'total_charges': [846, 2157.6, 179.4, 3596.4],
    'churn': [0, 0, 1, 0]
}
df = pd.DataFrame(data)

# Training parameters
params = {
    "model_type": "classification",
    "selected_model": "random_forest",
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 5,
        "min_samples_split": 2
    }
}

# Expected evaluation metrics
{
    "Accuracy": 0.92,
    "Precision": 0.91,
    "Recall": 0.93,
    "F1 Score": 0.92
}
```

### 3. AutoML Example: Automatic Model Selection

```python
# Example dataset structure
import pandas as pd
data = {
    'feature1': [1.2, 2.3, 3.4, 4.5],
    'feature2': [5.6, 6.7, 7.8, 8.9],
    'target': [10, 20, 30, 40]
}
df = pd.DataFrame(data)

# Training parameters
params = {
    "model_type": "regression",
    "selected_model": "auto",
    "test_size": 0.2
}

# The system will automatically:
# 1. Try different models
# 2. Perform hyperparameter tuning
# 3. Select the best performing model
# 4. Return the best model and its metrics
```

## Project Structure

```
├── app.py                 # Main application file
├── requirements.txt       # Project dependencies
├── uploads/              # Directory for uploaded datasets
├── models/              # Directory for saved models
└── templates/           # HTML templates
    ├── main.html
    ├── regression.html
    └── classification.html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flask for the web framework
- scikit-learn for machine learning algorithms
- XGBoost, CatBoost, and LightGBM teams for their gradient boosting implementations
- ONNX community for model interoperability standards 