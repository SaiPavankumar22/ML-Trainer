from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, send_file, session
import os
import pickle
import joblib
import dill
import base64
import pandas as pd
import numpy as np
import logging
from werkzeug.utils import secure_filename
from flask_swagger_ui import get_swaggerui_blueprint
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import traceback

app = Flask(__name__)
app.secret_key = 'your-secret-key'
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
val=""
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Import ML libraries
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from onnxmltools.convert import convert_xgboost

# Import sklearn and other basic ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import zscore

# Lazy loading of ML libraries
def import_ml_libraries():
    global tensorflow, torch, onnx, coremltools
    import tensorflow as tf
    import torch
    import onnx
    import coremltools as ct

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_target_column(df, target_column):
    """Perform comprehensive analysis of the target column."""
    try:
        n_samples = len(df)
        n_unique = df[target_column].nunique()
        null_ratio = df[target_column].isnull().sum() / n_samples
        is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
        
        analysis = {
            'n_samples': n_samples,
            'n_unique': n_unique,
            'null_ratio': null_ratio,
            'is_numeric': is_numeric,
            'dtype': str(df[target_column].dtype)
        }
        
        if is_numeric:
            analysis.update({
                'mean': df[target_column].mean(),
                'std': df[target_column].std(),
                'min': df[target_column].min(),
                'max': df[target_column].max(),
                'median': df[target_column].median(),
                'skewness': df[target_column].skew(),
                'kurtosis': df[target_column].kurtosis()
            })
            
            Q1 = df[target_column].quantile(0.25)
            Q3 = df[target_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)]
            analysis['outlier_ratio'] = len(outliers) / n_samples
        else:
            analysis.update({
                'avg_length': df[target_column].str.len().mean(),
                'pattern_ratio': df[target_column].str.match(r'^[A-Za-z]+$').mean()
            })
        
        return analysis
    except Exception as e:
        logger.error(f"Error in analyze_target_column: {str(e)}")
        return None

def determine_task_type(df, target_column, analysis=None):
    """Determine if the task is classification or regression with confidence score."""
    try:
        if analysis is None:
            analysis = analyze_target_column(df, target_column)
        
        if analysis is None:
            return "regression", 0.5
        
        confidence = 0.5
        
        if analysis['null_ratio'] > 0:
            if analysis['is_numeric']:
                df[target_column] = df[target_column].fillna(df[target_column].median())
            else:
                df[target_column] = df[target_column].fillna(df[target_column].mode()[0])
        
        if analysis['is_numeric']:
            unique_values = df[target_column].unique()
            if len(unique_values) == 2:
                if set(unique_values) in [{0, 1}, {-1, 1}]:
                    confidence = 0.95
                    return "classification", confidence
            
            if analysis['n_unique'] <= 10 and analysis['n_unique'] > 2:
                if all(x.is_integer() for x in unique_values if pd.notnull(x)):
                    confidence = 0.85
                    return "classification", confidence
            
            # Check for regression
            if analysis['n_unique'] > 10:
                confidence = 0.8
                return "regression", confidence
        else:
            # For non-numeric data, it's likely classification
            confidence = 0.9
            return "classification", confidence
        
        # Adjust confidence based on sample size
        if analysis['n_samples'] < 100:
            confidence *= 0.8
        
        # Adjust confidence based on outliers
        if analysis.get('outlier_ratio', 0) > 0.1:
            confidence *= 0.9
        
        return "regression", confidence  # Default to regression if no clear decision
    except Exception as e:
        logger.error(f"Error in determine_task_type: {str(e)}")
        return "regression", 0.5

def save_and_encode(file_path, save_func, *args):
    with open(file_path, "wb") as f:
        save_func(*args, f)
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return encoded

regression_models = {
    "simple_linear": LinearRegression(),
    "multiple_linear": LinearRegression(),
    "ridge": Ridge(),
    "lasso": Lasso(),
    "elastic_net": ElasticNet(),
    "svr": SVR(),
    "decision_tree": DecisionTreeRegressor(),
    "random_forest": RandomForestRegressor(),
    "xgboost": XGBRegressor(),
    "gradient_boosting": GradientBoostingRegressor(),
    "adaboost": AdaBoostRegressor(),
    # CatBoost often performs well because it:
    # 1. Handles categorical features automatically without one-hot encoding
    # 2. Uses ordered boosting vs. traditional gradient boosting
    # 3. Reduces overfitting with a combination of techniques 
    # 4. The "learn", "total", "remaining" outputs show its training progress per iteration
    "catboost": CatBoostRegressor(verbose=1)
}

classification_models = {
    "logistic": LogisticRegression(),
    "ridge_classifier": RidgeClassifier(),
    "svc_linear": SVC(kernel="linear"),
    "svc_rbf": SVC(kernel="rbf"),
    "svc_poly": SVC(kernel="poly"),
    "svc_sigmoid": SVC(kernel="sigmoid"),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "gradient_boosting": GradientBoostingClassifier(),
    "adaboost": AdaBoostClassifier(),
    "xgboost": XGBClassifier(),
    # CatBoost often performs well because it:
    # 1. Handles categorical features automatically without one-hot encoding
    # 2. Uses ordered boosting vs. traditional gradient boosting
    # 3. Reduces overfitting with a combination of techniques
    # 4. The "learn", "total", "remaining" outputs show its training progress per iteration
    "catboost": CatBoostClassifier(verbose=1),
    "gaussian_nb": GaussianNB(),
    "multinomial_nb": MultinomialNB(),
    "bernoulli_nb": BernoulliNB(),
    "complement_nb": ComplementNB(),
    "knn": KNeighborsClassifier()
}

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/output')
def output():
    return render_template('predictions.html')


filename = None
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    if file.filename == '':
        return "No file selected"
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    val=file.filename
    df = pd.read_csv(file_path)
    target_column = df.iloc[:, -1]
    print("Redirecting with filename:", file.filename)
    
    processed_filename = f"processed_{file.filename}"
    
    if target_column.dtype == 'object' or target_column.nunique() < 20:
        return redirect(url_for('classification', filename=processed_filename))
    else:
        return redirect(url_for('regression', filename=processed_filename))

@app.route('/confirm_task_type', methods=['GET', 'POST'])
def confirm_task_type():
    if request.method == 'POST':
        task_type = request.form.get('task_type')
        filename = session['analysis']['filename']
        return redirect(url_for(task_type, filename=filename))
    
    analysis = session.get('analysis', {})
    return render_template('confirm_task_type.html', analysis=analysis)

@app.route('/regression/<filename>')
def regression(filename):
    print("Received filename:", filename)  
    return render_template('regression.html', filename=filename, models=regression_models.keys())

@app.route('/classification/<filename>')
def classification(filename):
    return render_template('classification.html', filename=filename, models=classification_models.keys())

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    if 'file' not in request.files or 'target_column' not in request.form:
        return {'error': 'File and target column name are required'}, 400

    file = request.files['file']
    target_column = request.form['target_column']
    # Store target column in session
    session['target_column'] = target_column
    print("Received target column:", target_column) 
    print("Received filename:", file.filename)

    if file.filename == '':
        return {'error': 'No selected file'}, 400

    if not allowed_file(file.filename):
        return {'error': 'Invalid file format. Only CSV or Excel allowed.'}, 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
    except Exception as e:
        return {'error': f'Error reading file: {str(e)}'}, 500

    # Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Ensure the target column exists
    if target_column not in df.columns:
        return {'error': f'Target column "{target_column}" not found in dataset'}, 400

    # Identify numerical and categorical columns (excluding target column)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)
    if target_column in categorical_cols:
        categorical_cols.remove(target_column)

    # Handle missing values
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

    if categorical_cols:
        mode_values = df[categorical_cols].mode()
        if not mode_values.empty:
            df[categorical_cols] = df[categorical_cols].fillna(mode_values.iloc[0])

    # Handle outliers using Z-score method (removes extreme values)
    z_scores = np.abs(zscore(df[numerical_cols]))
    df = df[(z_scores < 3).all(axis=1)]

    # Encode categorical features (One-Hot for Nominal, Label Encoding for Ordinal)
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    for col in categorical_cols:
        if col == target_column:
            continue  # Skip the target column
        if df[col].nunique() > 2:  # Nominal Data
            encoded_df = pd.DataFrame(encoder.fit_transform(df[[col]]), 
                                  columns=encoder.get_feature_names_out([col]),
                                  index=df.index)
            df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
        else:
            df[col] = LabelEncoder().fit_transform(df[col])

    # Apply scaling **only if necessary**
    def needs_scaling(series):
        return series.max() - series.min() > 1000  # Scale only large-range columns

    scaler = StandardScaler() if len(df) > 1000 else MinMaxScaler()

    for col in numerical_cols:
        if col == target_column:
            continue  # Do not scale the target column
        if needs_scaling(df[col]):
            df[[col]] = scaler.fit_transform(df[[col]])

    # Drop any empty rows after transformations
    df.dropna(how='all', inplace=True)

    processed_filename = f"processed_{filename}"
    processed_filepath = os.path.join(UPLOAD_FOLDER, processed_filename)
    df.to_csv(processed_filepath, index=False)

    response = send_file(processed_filepath, as_attachment=True)
    response.headers['X-Processed-Filename'] = processed_filename
    return response

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        print(data)
        filename = data['filename']
        model_type = data['model_type']
        selected_model = data['selected_model']
        test_size = float(data['test_size'])
        hyperparameters = data['hyperparameters']
        
        df = pd.read_csv(os.path.join(UPLOAD_FOLDER, filename))
        
        # Check for NaN values and handle them
        if df.isnull().any().any():
            print("Warning: Found NaN values in the dataset, filling them")
            df = df.fillna(df.mean())
        
        X = df.iloc[:, :-1]  # Features (all columns except the last)
        y = df.iloc[:, -1]   # Target (last column)

        # --- Encode all categorical features and save encoders ---
        feature_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object' or not np.issubdtype(X[col].dtype, np.number):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                feature_encoders[col] = le
        
        # Save feature encoders
        if feature_encoders:
            encoders_path = os.path.join(MODEL_FOLDER, f"{filename.split('.')[0]}_feature_encoders.pkl")
            with open(encoders_path, 'wb') as f:
                pickle.dump(feature_encoders, f)

        # --- Encode target if it's not numeric ---
        label_encoder = None
        if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            # Save the encoder for later use
            encoder_path = os.path.join(MODEL_FOLDER, f"{filename.split('.')[0]}_label_encoder.pkl")
            with open(encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Set up models
        if model_type == 'regression':
            models = regression_models
        else:
            models = classification_models
            
        # Train model
        if selected_model == "auto":
            # Use our improved auto_train function
            best_model = auto_train(models, X_train, y_train, model_type)
            if best_model is None:
                return jsonify({
                    "error": "Failed to train any model. Please check your data and try again."
                }), 400
        else:
            try:
                # Handle specific model training
                model = models[selected_model]
                
                # Process hyperparameters
                key_replacements = {
                    "c_value": "C",
                    "Max Iterations": "max_iter",
                    "max_iterations": "max_iter"
                }
                criterion_replacements = {
                    "mse": "squared_error",
                    "mae": "absolute_error"
                }
                
                updated_hyperparameters = {}
                for key, value in hyperparameters.items():
                    new_key = key_replacements.get(key, key)
                    if new_key == "criterion" and value in criterion_replacements:
                        value = criterion_replacements[value]
                    if new_key == "C":
                        value = float(value)
                    if isinstance(value, str) and (value.isnumeric() or value in ["True", "False", "None"]):
                        value = eval(value)
                    updated_hyperparameters[new_key] = value
                
                model.set_params(**updated_hyperparameters)
                
                # Convert to numpy arrays if needed
                X_train_array = X_train.values if hasattr(X_train, 'values') else X_train
                y_train_array = y_train.values if hasattr(y_train, 'values') else y_train
                
                # Special handling for Naive Bayes models that can't handle negative values
                if selected_model in ["multinomial_nb", "complement_nb", "bernoulli_nb"]:
                    # For NB models that need non-negative values, use MinMaxScaler
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    X_train_array = scaler.fit_transform(X_train_array)
                    X_test = scaler.transform(X_test.values if hasattr(X_test, 'values') else X_test)
                
                # Train the model
                model.fit(X_train_array, y_train_array)
                best_model = model
            except Exception as e:
                print(f"Error training model: {str(e)}")
                return jsonify({
                    "error": f"Error training model: {str(e)}"
                }), 400
                
        # Generate predictions for evaluation
        try:
            X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
            y_pred = best_model.predict(X_test_array)
            
            # Evaluate model
            if model_type == "regression":
                evaluation_metrics = {
                    "MSE": float(mean_squared_error(y_test, y_pred)),
                    "RMSE": float(root_mean_squared_error(y_test, y_pred)),
                    "R2 Score": float(r2_score(y_test, y_pred))
                }
                
                # Check for overfitting in regression models
                r2_train = float(best_model.score(X_train.values if hasattr(X_train, 'values') else X_train, 
                                          y_train.values if hasattr(y_train, 'values') else y_train))
                r2_test = float(evaluation_metrics["R2 Score"])
                
                # If R2 score is too close to 1.0 or there's a big gap between train and test
                if r2_train > 0.99:
                    warnings = ["Warning: The model shows signs of overfitting with nearly perfect training score (R² > 0.99)."]
                elif r2_train - r2_test > 0.3:  # If the drop is more than 0.3
                    warnings = [f"Warning: The model may be overfitting. Training R² ({r2_train:.4f}) is significantly higher than Test R² ({r2_test:.4f})."]
                else:
                    warnings = []
                     
            else:
                evaluation_metrics = {
                    "Accuracy": float(accuracy_score(y_test, y_pred)),
                    "Precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "Recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                    "F1 Score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
                }
                
                # Check for overfitting in classification models
                acc_train = float(best_model.score(X_train.values if hasattr(X_train, 'values') else X_train, 
                                           y_train.values if hasattr(y_train, 'values') else y_train))
                acc_test = float(evaluation_metrics["Accuracy"])
                
                # If accuracy is too close to 1.0 or there's a big gap between train and test
                if acc_train > 0.99:
                    warnings = ["Warning: The model shows signs of overfitting with nearly perfect training accuracy (> 99%)."]
                elif acc_train - acc_test > 0.2:  # If the drop is more than 20%
                    warnings = [f"Warning: The model may be overfitting. Training accuracy ({acc_train:.4f}) is significantly higher than Test accuracy ({acc_test:.4f})."]
                else:
                    warnings = []
        except Exception as e:
            print(f"Error evaluating model: {str(e)}")
            return jsonify({
                "error": f"Error evaluating model: {str(e)}"
            }), 400
            
        # Save model in different formats
        base_filename = os.path.join(MODEL_FOLDER, filename.split('.')[0])
        model_files = {}
        
        try:
            model_files["pkl"] = save_and_encode(f"{base_filename}.pkl", pickle.dump, best_model)
            model_files["joblib"] = save_and_encode(f"{base_filename}.joblib", joblib.dump, best_model)
            model_files["sav"] = save_and_encode(f"{base_filename}.sav", pickle.dump, best_model)
            model_files["dill"] = save_and_encode(f"{base_filename}.dill", dill.dump, best_model)
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            # Continue even if some formats fail to save
            
        # Try to convert to ONNX if possible
        try:
            # Only attempt ONNX conversion for sklearn models (not XGBoost or CatBoost)
            is_onnx_compatible = not type(best_model).__name__.startswith(('XGB', 'CatBoost'))
            
            if is_onnx_compatible:
                # Use standard skl2onnx conversion for other models
                initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
                onnx_model = convert_sklearn(best_model, initial_types=initial_type)
                model_files["onnx"] = save_and_encode(f"{base_filename}.onnx", lambda f: f.write(onnx_model.SerializeToString()))
            else:
                print(f"Skipping ONNX conversion for {type(best_model).__name__} as it's not directly supported")
                model_files["onnx"] = None
        except Exception as e:
            print(f"Warning: Failed to convert model to ONNX: {str(e)}")
            model_files["onnx"] = None
            
        # Get the model's name and hyperparameters for display
        if selected_model == "auto":
            try:
                # Get model type name first
                best_model_name = type(best_model).__name__
                
                # Extract hyperparameters directly from the model's params
                if hasattr(best_model, 'get_params'):
                    model_params = best_model.get_params()
                    # Convert all values to JSON-serializable types and remove None values
                    hyperparameters = {}
                    for k, v in model_params.items():
                        if v is not None:
                            # Convert numpy types to Python native types
                            if isinstance(v, (np.integer, np.floating, np.bool_)):
                                hyperparameters[k] = v.item()
                            elif isinstance(v, np.ndarray):
                                hyperparameters[k] = v.tolist()
                            elif pd.isna(v):  # Handle NaN values
                                hyperparameters[k] = None
                            else:
                                # Try to convert to a basic type, otherwise use string representation
                                try:
                                    json.dumps({k: v})  # Test if it's JSON serializable
                                    hyperparameters[k] = v
                                except (TypeError, OverflowError):
                                    hyperparameters[k] = str(v)
                else:
                    hyperparameters = {}
            except Exception as e:
                print(f"Error getting model info: {str(e)}")
                hyperparameters = {}
        else:
            best_model_name = selected_model
            
        # Return results with warnings if needed
        response = {
            "selected_model": best_model_name,
            "hyperparameters": hyperparameters,
            "evaluation_metrics": evaluation_metrics,
            "model_files": model_files
        }
        
        # Add warnings if there are any
        if 'warnings' in locals() and warnings:
            response["warnings"] = warnings
        
        # Ensure response is JSON serializable
        try:
            # Test JSON serialization before returning
            json.dumps(response)
        except (TypeError, OverflowError) as e:
            print(f"JSON serialization error: {str(e)}")
            # Clean the response to make it serializable
            cleaned_response = {
                "selected_model": str(response["selected_model"]),
                "hyperparameters": {k: str(v) if not isinstance(v, (bool, int, float, str, list, dict, type(None))) else v 
                                   for k, v in response["hyperparameters"].items()},
                "evaluation_metrics": response["evaluation_metrics"],
                "model_files": response["model_files"]
            }
            if "warnings" in response:
                cleaned_response["warnings"] = response["warnings"]
            return jsonify(cleaned_response)
        
        return jsonify(response)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            "error": f"Unexpected error: {str(e)}"
        }), 500

def auto_train(models, X_train, y_train, model_type):
    best_model = None
    best_score = float('-inf') 
    param_grids = {
    "simple_linear": {"fit_intercept": [True, False]},
    "multiple_linear": {"fit_intercept": [True, False]},
    "ridge": {"alpha": [0.01, 0.1, 1, 10]},
    "lasso": {"alpha": [0.01, 0.1, 1]},
    "elastic_net": {"alpha": [0.01, 0.1, 1], "l1_ratio": [0.2, 0.5, 0.8]},
    "svr": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "decision_tree": {"max_depth": [None, 5, 10, 15], "min_samples_split": [2, 5, 10]},
    "random_forest": {"n_estimators": [50, 100], "max_depth": [None, 5, 10, 15]},
    "xgboost": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]},
    "gradient_boosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5]},
    "adaboost": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 1]},
    "catboost": {"depth": [4, 6, 8], "learning_rate": [0.01, 0.1], "iterations": [50, 100]},
    "logistic": {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    "ridge_classifier": {"alpha": [0.1, 1, 10]},
    "svc_linear": {"C": [0.1, 1, 10]},
    "svc_rbf": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "svc_poly": {"C": [1, 10], "degree": [2, 3]},
    "svc_sigmoid": {"C": [0.1, 1, 10], "gamma": ["scale"]},
    "decision_tree": {"max_depth": [None, 5, 10, 15], "min_samples_split": [2, 5, 10], "criterion": ["gini", "entropy"]},
    "random_forest": {"n_estimators": [50, 100], "max_depth": [None, 5, 10, 15], "criterion": ["gini", "entropy"]},
    "gradient_boosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5]},
    "adaboost": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 1]},
    "xgboost": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]},
    "catboost": {"depth": [4, 6, 8], "learning_rate": [0.01, 0.1], "iterations": [50, 100]},
    "gaussian_nb": {},
    "multinomial_nb": {"alpha": [0.1, 0.5, 1.0]},
    "bernoulli_nb": {"alpha": [0.1, 0.5, 1.0]},
    "complement_nb": {"alpha": [0.1, 0.5, 1.0]},
    "knn": {"n_neighbors": [3, 5, 7, 10], "weights": ["uniform", "distance"]}
}
    print("Starting auto-training...")
    print(f"Available models: {models.keys()}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Target data shape: {y_train.shape}")
    
    # Convert pandas DataFrame to numpy array if needed
    if hasattr(X_train, 'values'):
        print("Converting pandas DataFrame to numpy array")
        X_train_array = X_train.values
    else:
        X_train_array = X_train
        
    if hasattr(y_train, 'values'):
        y_train_array = y_train.values
    else:
        y_train_array = y_train
    
    # Create a scaled version of the data for models that need non-negative values
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_array)
    
    # Setup cross-validation for more reliable model selection
    from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
    if model_type == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Prioritize models based on task type - evaluate high-performing models first
    priority_models = []
    other_models = []
    
    # For classification tasks, prioritize tree-based models and ensemble methods
    if model_type == 'classification':
        high_priority = ["catboost", "xgboost", "random_forest", "gradient_boosting"]
        medium_priority = ["decision_tree", "logistic", "svc_rbf", "adaboost"]
        
        for name in models.keys():
            if name in high_priority:
                priority_models.append(name)
            elif name in medium_priority:
                other_models.insert(0, name)  # Add to beginning
            else:
                other_models.append(name)  # Add to end
    else:
        # For regression, use a similar but adjusted prioritization
        high_priority = ["catboost", "xgboost", "random_forest", "gradient_boosting"]
        medium_priority = ["elastic_net", "ridge", "svr", "decision_tree"]
        
        for name in models.keys():
            if name in high_priority:
                priority_models.append(name)
            elif name in medium_priority:
                other_models.insert(0, name)
            else:
                other_models.append(name)
    
    model_order = priority_models + other_models
    print(f"Model evaluation order: {model_order}")
    
    # Try models with more thorough evaluation
    for model_name in model_order:
        model = models[model_name]
        
        # Get a clone of the model to avoid changing the original
        from sklearn.base import clone
        print(f"\n{'='*20} Testing {model_name} {'='*20}")
        
        # Handle special case for CatBoost - temporarily disable verbose output during CV
        is_catboost = model_name == "catboost"
        original_verbose = None
        if is_catboost and hasattr(model, 'get_params') and 'verbose' in model.get_params():
            original_verbose = model.get_params()['verbose']
            model.set_params(verbose=0)
        
        param_grid = param_grids.get(model_name, {})

        if not param_grid:
            print(f"⚠️ Warning: No parameters found for {model_name}, using default parameters")
            try:
                # Choose appropriate data based on model type
                if model_name in ["multinomial_nb", "complement_nb", "bernoulli_nb"]:
                    # These models require non-negative data
                    current_model = clone(model)
                    # Always use cross-validation, even for models without hyperparameters
                    try:
                        current_model.fit(X_train_scaled, y_train_array)
                        # Use cross-validation to get a more reliable score
                        cv_scores = cross_val_score(
                            current_model, X_train_scaled, y_train_array, 
                            cv=cv, scoring='accuracy' if model_type == 'classification' else 'r2'
                        )
                        score = cv_scores.mean()
                        print(f"✅ {model_name} CV Score: {score:.4f} (std: {cv_scores.std():.4f})")
                    except Exception as e:
                        print(f"  - Cross-validation failed for {model_name}, falling back to direct score: {str(e)}")
                        current_model.fit(X_train_scaled, y_train_array)
                        score = current_model.score(X_train_scaled, y_train_array)
                        print(f"⚠️ {model_name} Training Score: {score:.4f} (not cross-validated)")
                else:
                    # Regular models use standard data
                    current_model = clone(model)
                    try:
                        current_model.fit(X_train_array, y_train_array)
                        # Use cross-validation to get a more reliable score
                        cv_scores = cross_val_score(
                            current_model, X_train_array, y_train_array, 
                            cv=cv, scoring='accuracy' if model_type == 'classification' else 'r2'
                        )
                        score = cv_scores.mean()
                        print(f"✅ {model_name} CV Score: {score:.4f} (std: {cv_scores.std():.4f})")
                    except Exception as e:
                        print(f"  - Cross-validation failed for {model_name}, falling back to direct score: {str(e)}")
                        current_model.fit(X_train_array, y_train_array)
                        score = current_model.score(X_train_array, y_train_array)
                        print(f"⚠️ {model_name} Training Score: {score:.4f} (not cross-validated)")
                
                if score > best_score:
                    best_model = current_model
                    best_score = score
                    print(f"🎯 New best model found: {model_name} with score {score:.4f}")
            except Exception as e:
                print(f"❌ Error training {model_name} with default parameters: {str(e)}")
            continue

        try:
            print(f"Training {model_name} with parameters {param_grid}")
            
            # More thorough parameter search with cross-validation
            best_param_score = -float('inf')
            best_params = None
            best_param_model = None
            
            # Try each parameter combination
            import itertools
            import numpy as np
            
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            
            # If too many combinations, sample a subset
            param_combinations = list(itertools.product(*param_values))
            # Use a larger number of combinations for important models like CatBoost
            max_combinations = 30 if model_name in ["catboost", "xgboost", "random_forest"] else 20
            if len(param_combinations) > max_combinations:
                import random
                random.seed(42)
                param_combinations = random.sample(param_combinations, max_combinations)
                print(f"  - Sampling {max_combinations} parameter combinations from {len(list(itertools.product(*param_values)))} total")
            
            for param_combo in param_combinations:
                param_dict = dict(zip(param_names, param_combo))
                
                # For CatBoost, ensure verbose is set to 0 during cross-validation
                if is_catboost:
                    param_dict['verbose'] = 0
                
                try:
                    # Create a fresh copy of the model for each parameter combination
                    param_model = clone(model)
                    
                    # Set parameters
                    param_model.set_params(**param_dict)
                    
                    # Choose appropriate data based on model type
                    if model_name in ["multinomial_nb", "complement_nb", "bernoulli_nb"]:
                        # These models require non-negative data
                        # Use cross-validation for more reliable scoring
                        try:
                            # For NB models, fixed random seed may not be enough, so we'll fit once first
                            param_model.fit(X_train_scaled, y_train_array)
                            cv_scores = cross_val_score(param_model, X_train_scaled, y_train_array, cv=cv, 
                                                       scoring='r2' if model_type == 'regression' else 'accuracy')
                            cv_score = np.mean(cv_scores)
                            print(f"  - {model_name} with {param_dict}: CV Score = {cv_score:.4f}")
                            
                            if cv_score > best_param_score:
                                best_param_score = cv_score
                                best_params = param_dict.copy()
                                # Train a fresh model with these params for final use
                                best_param_model = clone(model)
                                best_param_model.set_params(**best_params)
                                best_param_model.fit(X_train_scaled, y_train_array)
                        except Exception as e:
                            print(f"  - Cross-val failed for {param_dict}: {str(e)}")
                            # Fallback to direct fitting if cross-validation fails
                            param_model.fit(X_train_scaled, y_train_array)
                            param_score = param_model.score(X_train_scaled, y_train_array)
                            if param_score > best_param_score:
                                best_param_score = param_score
                                best_params = param_dict.copy()
                                best_param_model = param_model
                    else:
                        # Regular models use standard data with cross-validation
                        try:
                            cv_scores = cross_val_score(param_model, X_train_array, y_train_array, cv=cv, 
                                                      scoring='r2' if model_type == 'regression' else 'accuracy')
                            cv_score = np.mean(cv_scores)
                            print(f"  - {model_name} with {param_dict}: CV Score = {cv_score:.4f}")
                            
                            if cv_score > best_param_score:
                                best_param_score = cv_score
                                best_params = param_dict.copy()
                                # Train a fresh model with these params for final use
                                best_param_model = clone(model)
                                best_param_model.set_params(**best_params)
                                best_param_model.fit(X_train_array, y_train_array)
                        except Exception as e:
                            print(f"  - Cross-val failed for {param_dict}: {str(e)}")
                            # Fallback to direct fitting if cross-validation fails
                            param_model.fit(X_train_array, y_train_array)
                            # Calculate score
                            if model_type == 'regression':
                                param_score = param_model.score(X_train_array, y_train_array)  # R^2 for regression
                            else:
                                param_score = param_model.score(X_train_array, y_train_array)  # Accuracy for classification
                            
                            if param_score > best_param_score:
                                best_param_score = param_score
                                best_params = param_dict.copy()
                                best_param_model = param_model
                except Exception as e:
                    print(f"  - Failed with params {param_dict}: {str(e)}")
                    continue
            
            if best_params is not None:
                print(f"✅ {model_name} Best Score: {best_param_score:.4f}")
                print(f"Best parameters: {best_params}")
                
                # Get the final score using cross-validation if possible
                try:
                    if model_name in ["multinomial_nb", "complement_nb", "bernoulli_nb"]:
                        # For NB models that need non-negative values
                        cv_scores = cross_val_score(best_param_model, X_train_scaled, y_train_array, cv=cv,
                                                   scoring='r2' if model_type == 'regression' else 'accuracy')
                    else:
                        cv_scores = cross_val_score(best_param_model, X_train_array, y_train_array, cv=cv,
                                                  scoring='r2' if model_type == 'regression' else 'accuracy') 
                    final_score = np.mean(cv_scores)
                    print(f"Final CV score: {final_score:.4f}")
                except Exception as e:
                    print(f"  - Cross-validation failed for final score: {str(e)}")
                    # Fallback to training score
                    if model_name in ["multinomial_nb", "complement_nb", "bernoulli_nb"]:
                        final_score = best_param_model.score(X_train_scaled, y_train_array)
                    else:
                        final_score = best_param_model.score(X_train_array, y_train_array)
                    print(f"⚠️ Final training score (not CV): {final_score:.4f}")
                
                if best_param_score > best_score:
                    best_model = best_param_model
                    best_score = best_param_score
                    print(f"🎯 New best model found: {model_name} with CV score {best_param_score:.4f}")
            else:
                print(f"❌ Could not find working parameters for {model_name}")

        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            continue
        
        # Restore original verbose setting for CatBoost if we changed it
        if is_catboost and original_verbose is not None and best_model is not None and model_name == type(best_model).__name__:
            if hasattr(best_model, 'set_params'):
                best_model.set_params(verbose=original_verbose)

    if best_model is None:
        print("\n❌ Auto-training failed: No valid model found.")
        print("Attempting fallback to simple model...")
        try:
            if model_type == 'regression':
                print("Trying LinearRegression as fallback")
                fallback_model = LinearRegression()
            else:
                print("Trying LogisticRegression as fallback")
                fallback_model = LogisticRegression(max_iter=1000)
            
            fallback_model.fit(X_train_array, y_train_array)
            best_model = fallback_model
            print("✅ Fallback model trained successfully")
        except Exception as e:
            print(f"❌ Fallback model also failed: {str(e)}")
    else:
        print(f"\n🏆 Best Model Selected: {type(best_model).__name__} with Score: {best_score:.4f}")
        
        # For CatBoost, restore original verbose setting before returning
        if is_catboost and original_verbose is not None:
            if type(best_model).__name__ == "CatBoostClassifier" or type(best_model).__name__ == "CatBoostRegressor":
                best_model.set_params(verbose=original_verbose)

    return best_model

MODEL_FOLDER = os.path.join(os.path.dirname(__file__), "models")

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    filename = secure_filename(filename)
    filepath = os.path.join(MODEL_FOLDER, filename)
    print(f"[DEBUG] Looking for: {filepath}")

    if not os.path.exists(filepath):
        print("[DEBUG] File not found.")
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(MODEL_FOLDER, filename, as_attachment=True)

def get_model_features(filename, target_column=None):
    """Get feature names from the dataset used to train the model."""
    try:
        logger.info(f"Getting model features for file: {filename}")
        logger.info(f"Target column provided: {target_column}")
        
        # Read the processed dataset
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        logger.info(f"Reading file from: {file_path}")
        df = pd.read_csv(file_path)
        
        logger.info(f"Dataset columns: {df.columns.tolist()}")
        
        # If target_column is provided, exclude it from features
        if target_column and target_column in df.columns:
            features = [col for col in df.columns if col != target_column]
            logger.info(f"Features after excluding target column: {features}")
            return features
        
        # Fallback to old behavior if target_column is not provided
        logger.warning("No target column provided, falling back to excluding last column")
        features = df.columns[:-1].tolist()
        logger.info(f"Features using fallback method: {features}")
        return features
    except Exception as e:
        logger.error(f"Error getting model features: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

@app.route('/predictions/<filename>')
def predictions(filename):
    """Render the predictions page with available models and features."""
    logger.info(f"Predictions route called for file: {filename}")
    
    # Get target column from session if available
    target_column = session.get('target_column')
    logger.info(f"Target column from session: {target_column}")
    logger.info(f"Session contents: {dict(session)}")
    
    features = get_model_features(filename, target_column)
    logger.info(f"Features returned for predictions page: {features}")
    
    return render_template('predictions.html', filename=filename, features=features)

@app.route('/predict_value', methods=['POST'])
def predict_value():
    """Handle prediction requests."""
    try:
        data = request.json
        logger.info(f"Received prediction request data: {data}")
        
        filename = data['filename']
        model_format = data['model_format']
        input_values = data['input_values']
        
        logger.info(f"Processing prediction for file: {filename}")
        logger.info(f"Model format: {model_format}")
        logger.info(f"Input values: {input_values}")
        
        # Load the appropriate model based on format
        model_path = os.path.join(MODEL_FOLDER, f"{filename.split('.')[0]}.{model_format}")
        logger.info(f"Looking for model at: {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return jsonify({"error": f"Model file not found: {model_path}"}), 404
            
        # Load model based on format
        try:
            if model_format == 'pkl':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif model_format == 'joblib':
                model = joblib.load(model_path)
            elif model_format == 'sav':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif model_format == 'dill':
                with open(model_path, 'rb') as f:
                    model = dill.load(f)
            elif model_format == 'onnx':
                return jsonify({"error": "ONNX models are not supported for direct prediction. Please use a different format (pkl, joblib, sav, or dill)."}), 400
            else:
                return jsonify({"error": f"Unsupported model format: {model_format}. Please use pkl, joblib, sav, or dill."}), 400
            logger.info(f"Successfully loaded model of type: {type(model)}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": f"Error loading model: {str(e)}"}), 500
            
        # Load feature encoders if they exist
        encoders_path = os.path.join(MODEL_FOLDER, f"{filename.split('.')[0]}_feature_encoders.pkl")
        feature_encoders = {}
        if os.path.exists(encoders_path):
            with open(encoders_path, 'rb') as f:
                feature_encoders = pickle.load(f)
            logger.info(f"Loaded feature encoders for columns: {list(feature_encoders.keys())}")
        
        # Try to load label encoder for the target
        encoder_path = os.path.join(MODEL_FOLDER, f"{filename.split('.')[0]}_label_encoder.pkl")
        label_encoder = None
        if os.path.exists(encoder_path):
            with open(encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            logger.info("Successfully loaded label encoder")
        
        # Convert input values to numeric array, using feature encoders if available
        try:
            input_array = []
            for feature in input_values.keys():
                value = input_values[feature]
                if feature in feature_encoders:
                    value = feature_encoders[feature].transform([value])[0]
                input_array.append(float(value))
            input_array = np.array(input_array).reshape(1, -1)
            logger.info(f"Processed input array shape: {input_array.shape}")
        except ValueError as e:
            logger.error(f"Invalid input value: {str(e)}")
            return jsonify({"error": f"Invalid input value: {str(e)}. All inputs must be numeric or valid class labels."}), 400
        except Exception as e:
            logger.error(f"Error processing input values: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": f"Error processing input values: {str(e)}"}), 500
        
        # Make prediction
        try:
            prediction = model.predict(input_array)
            logger.info(f"Raw prediction: {prediction}")
            
            # Try to decode prediction if label encoder exists
            if label_encoder is not None:
                prediction = label_encoder.inverse_transform([int(prediction[0])])
                prediction = prediction[0]
                logger.info(f"Decoded prediction using label encoder: {prediction}")
            else:
                prediction = float(prediction[0])
                logger.info(f"Final prediction value: {prediction}")
            
            # If it's a classification model, get probabilities if available
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(input_array)
                    logger.info(f"Raw probabilities: {probabilities}")
                    
                    if label_encoder is not None:
                        class_labels = label_encoder.inverse_transform(np.arange(probabilities.shape[1]))
                        result = {
                            "prediction": prediction,
                            "probabilities": dict(zip(class_labels, probabilities[0].tolist()))
                        }
                    else:
                        result = {
                            "prediction": prediction,
                            "probabilities": probabilities[0].tolist()
                        }
                    logger.info(f"Final response with probabilities: {result}")
                    return jsonify(result)
                except Exception as e:
                    logger.error(f"Error getting probabilities: {str(e)}")
                    logger.error(f"Error type: {type(e).__name__}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return jsonify({
                        "prediction": prediction,
                        "warning": "Could not get class probabilities"
                    })
            
            logger.info(f"Final response: {{'prediction': {prediction}}}")
            return jsonify({"prediction": prediction})
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": f"Error making prediction: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in predict_value: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)