from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory, send_file, session
import os
import pickle
import joblib
import dill
import base64
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import onnx
import coremltools as ct
from werkzeug.utils import secure_filename
from flask_swagger_ui import get_swaggerui_blueprint
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from nyoka import skl_to_pmml
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from werkzeug.utils import secure_filename
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import logging
from onnxmltools.convert import convert_xgboost

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for session
UPLOAD_FOLDER = "uploads"
MODEL_FOLDER = "models"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
val=""
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_target_column(df, target_column):
    """Perform comprehensive analysis of the target column."""
    try:
        # Basic statistics
        n_samples = len(df)
        n_unique = df[target_column].nunique()
        null_ratio = df[target_column].isnull().sum() / n_samples
        
        # Check if column is numeric
        is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
        
        analysis = {
            'n_samples': n_samples,
            'n_unique': n_unique,
            'null_ratio': null_ratio,
            'is_numeric': is_numeric,
            'dtype': str(df[target_column].dtype)
        }
        
        if is_numeric:
            # Numeric-specific analysis
            analysis.update({
                'mean': df[target_column].mean(),
                'std': df[target_column].std(),
                'min': df[target_column].min(),
                'max': df[target_column].max(),
                'median': df[target_column].median(),
                'skewness': df[target_column].skew(),
                'kurtosis': df[target_column].kurtosis()
            })
            
            # Outlier detection using IQR
            Q1 = df[target_column].quantile(0.25)
            Q3 = df[target_column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[target_column] < lower_bound) | (df[target_column] > upper_bound)]
            analysis['outlier_ratio'] = len(outliers) / n_samples
        else:
            # Non-numeric analysis
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
            return "regression", 0.5  # Default fallback
        
        # Initialize confidence score
        confidence = 0.5
        
        # Handle missing values
        if analysis['null_ratio'] > 0:
            if analysis['is_numeric']:
                df[target_column] = df[target_column].fillna(df[target_column].median())
            else:
                df[target_column] = df[target_column].fillna(df[target_column].mode()[0])
        
        # Decision rules
        if analysis['is_numeric']:
            # Check for binary classification (0/1 or -1/1)
            unique_values = df[target_column].unique()
            if len(unique_values) == 2:
                if set(unique_values) in [{0, 1}, {-1, 1}]:
                    confidence = 0.95
                    return "classification", confidence
            
            # Check for multi-class classification
            if analysis['n_unique'] <= 10 and analysis['n_unique'] > 2:
                # Check if values are integers
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
        return "regression", 0.5  # Default fallback

def save_and_encode(file_path, save_func, *args):
    with open(file_path, "wb") as f:
        save_func(*args, f)  # Pass the file handle to the save function
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode('utf-8')
    return encoded


regression_models = {
    "simple_linear": LinearRegression(),
    "multiple_linear": LinearRegression(),
    "polynomial": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "ridge": Ridge(),
    "lasso": Lasso(),
    "elastic_net": ElasticNet(),
    "svr": SVR(),
    "decision_tree": DecisionTreeRegressor(),
    "random_forest": RandomForestRegressor(),
    "xgboost": XGBRegressor(),
    "gradient_boosting": GradientBoostingRegressor(),
    "adaboost": AdaBoostRegressor(),
    "catboost": CatBoostRegressor(verbose=0), 
    #"lightgbm": LGBMRegressor()
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
    #"lightgbm": LGBMClassifier(),
    "catboost": CatBoostClassifier(verbose=0),
    "gaussian_nb": GaussianNB(),
    "multinomial_nb": MultinomialNB(),
    "bernoulli_nb": BernoulliNB(),
    "complement_nb": ComplementNB(),
    "knn": KNeighborsClassifier()
}

@app.route('/')
def index():
    return render_template('main.html')
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
    
    # Create processed filename
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
        else:  # Ordinal Data
            df[col] = LabelEncoder().fit_transform(df[col])

    # Apply scaling **only if necessary**
    def needs_scaling(series):
        return series.max() - series.min() > 1000  # Scale only large-range columns

    # Choose scaler based on dataset size
    scaler = StandardScaler() if len(df) > 1000 else MinMaxScaler()

    # Apply scaling only to numerical columns excluding target
    for col in numerical_cols:
        if col == target_column:
            continue  # Do not scale the target column
        if needs_scaling(df[col]):
            df[[col]] = scaler.fit_transform(df[[col]])

    # Drop any empty rows after transformations
    df.dropna(how='all', inplace=True)

    # Save processed file
    processed_filename = f"processed_{filename}"
    processed_filepath = os.path.join(UPLOAD_FOLDER, processed_filename)
    df.to_csv(processed_filepath, index=False)

    # Return both the file and the processed filename
    response = send_file(processed_filepath, as_attachment=True)
    response.headers['X-Processed-Filename'] = processed_filename
    return response







@app.route('/train', methods=['POST'])
def train_model():
    data = request.json
    print(data)
    filename = data['filename']
    model_type = data['model_type']
    selected_model = data['selected_model']
    test_size = float(data['test_size'])
    hyperparameters = data['hyperparameters']
    
    df = pd.read_csv(os.path.join(UPLOAD_FOLDER, filename))
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    if model_type == 'regression':
        models = regression_models
    else:
        models = classification_models
    if selected_model == "auto":
        best_model = auto_train(models, X_train, y_train, model_type)
        if best_model is None:
            return jsonify({
                "error": "Failed to train any model. Please check your data and try again."
            }), 400
    else:
        model = models[selected_model]
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
        
        # For XGBoost models, rename features before training
        if selected_model == "xgboost":
            feature_names = [f'f{i}' for i in range(X_train.shape[1])]
            X_train = pd.DataFrame(X_train.values, columns=feature_names)
            X_test = pd.DataFrame(X_test.values, columns=feature_names)
        
        model.fit(X_train, y_train)
        best_model = model
    y_pred = best_model.predict(X_test)
    if model_type == "regression":
        evaluation_metrics = {
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred)
        }
    else:
        evaluation_metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
        }
    base_filename = os.path.join(MODEL_FOLDER, filename.split('.')[0])
    model_files = {}
    model_files["pkl"] = save_and_encode(f"{base_filename}.pkl", pickle.dump, best_model)
    model_files["joblib"] = save_and_encode(f"{base_filename}.joblib", joblib.dump, best_model)
    model_files["sav"] = save_and_encode(f"{base_filename}.sav", pickle.dump, best_model)
    model_files["dill"] = save_and_encode(f"{base_filename}.dill", dill.dump, best_model)
    
    # Special handling for XGBoost models
    if selected_model == "xgboost":
        try:
            # Convert XGBoost model to ONNX using onnxmltools
            initial_types = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
            onnx_model = convert_xgboost(best_model, initial_types=initial_types)
            model_files["onnx"] = save_and_encode(f"{base_filename}.onnx", lambda f: f.write(onnx_model.SerializeToString()))
        except Exception as e:
            print(f"Warning: Failed to convert XGBoost model to ONNX: {str(e)}")
            model_files["onnx"] = None
    else:
        try:
            # Use standard skl2onnx conversion for other models
            initial_type = [("float_input", FloatTensorType([None, X_train.shape[1]]))]
            onnx_model = convert_sklearn(best_model, initial_types=initial_type)
            model_files["onnx"] = save_and_encode(f"{base_filename}.onnx", lambda f: f.write(onnx_model.SerializeToString()))
        except Exception as e:
            print(f"Warning: Failed to convert model to ONNX: {str(e)}")
            model_files["onnx"] = None
    if selected_model == "auto":
        selected_model_str = str(best_model)  # Convert model object to string
        print("Auto-selected Model:", selected_model_str)

        if "(" in selected_model_str and ")" in selected_model_str:
            best_model_str, hyperparameters_str = selected_model_str.split("(", 1)  # Split only once
            best_model = best_model_str.strip()
    
            hyperparameters_str = hyperparameters_str.strip(")")
            hyperparameters = {}

            if hyperparameters_str:  # If there are hyperparameters
                try:
                    hyperparameters = {key.strip(): eval(value.strip()) for key, value in 
                                (item.split("=") for item in hyperparameters_str.split(","))}
                except Exception as e:
                    print("Error parsing hyperparameters:", e)
                    hyperparameters = {}
    else:
        best_model = selected_model
        hyperparameters = {}  # No hyperparameters if no parentheses


    #print(best_model)
    #print(hyperparameters)
    #print(evaluation_metrics)
    #print(model_files)
    return jsonify({
        "selected_model": best_model,
        "hyperparameters": hyperparameters,
        "evaluation_metrics": evaluation_metrics,
        "model_files": model_files
    })


def auto_train(models, X_train, y_train, model_type):
    best_model = None
    best_score = float('-inf') 
    param_grids = {
    #regression

    "simple_linear": {"fit_intercept": [True, False]},
    "multiple_linear": {"fit_intercept": [True, False]},
    "polynomial": {"polynomialfeatures__degree": [2, 3, 4]},
    "ridge": {"alpha": [0.01, 0.1, 1, 10]},
    "lasso": {"alpha": [0.01, 0.1, 1, 10]},
    "elastic_net": {"alpha": [0.01, 0.1, 1, 10], "l1_ratio": [0.1, 0.5, 0.9]},
    "svr": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf", "poly", "sigmoid"]},
    "decision_tree": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    "random_forest": {"n_estimators": [10, 50, 100], "max_depth": [3, 5, 10, None]},
    "xgboost": {"n_estimators": [10, 50, 100], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 10]},
    "gradient_boosting": {"n_estimators": [10, 50, 100], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 10]},
    "adaboost": {"n_estimators": [10, 50, 100], "learning_rate": [0.01, 0.1, 1]},
    "catboost": {"depth": [4, 6, 10], "learning_rate": [0.01, 0.1, 0.2], "iterations": [100, 500]},
    #"lightgbm": {"n_estimators": [10, 50, 100], "learning_rate": [0.01, 0.1, 0.2], "num_leaves": [10, 20, 40]},

    #classification models

    "logistic": {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
    "ridge_classifier": {"alpha": [0.01, 0.1, 1, 10]},
    "svc_linear": {"C": [0.1, 1, 10]},
    "svc_rbf": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "svc_poly": {"C": [0.1, 1, 10], "degree": [2, 3, 4]},
    "svc_sigmoid": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    "decision_tree": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
    "random_forest": {"n_estimators": [10, 50, 100], "max_depth": [3, 5, 10, None]},
    "gradient_boosting": {"n_estimators": [10, 50, 100], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 10]},
    "adaboost": {"n_estimators": [10, 50, 100], "learning_rate": [0.01, 0.1, 1]},
    "xgboost": {"n_estimators": [10, 50, 100], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 10]},
    #"lightgbm": {"n_estimators": [10, 50, 100], "learning_rate": [0.01, 0.1, 0.2], "num_leaves": [10, 20, 40]},
    "catboost": {"depth": [4, 6, 10], "learning_rate": [0.01, 0.1, 0.2], "iterations": [100, 500]},
    "gaussian_nb": {},  # No hyperparameters for GaussianNB
    "multinomial_nb": {"alpha": [0.01, 0.1, 1, 10]},
    "bernoulli_nb": {"alpha": [0.01, 0.1, 1, 10]},
    "complement_nb": {"alpha": [0.01, 0.1, 1, 10]},
    "knn": {"n_neighbors": [3, 5, 10], "weights": ["uniform", "distance"]}
}
    print("Starting auto-training...")
    print(f"Available models: {models.keys()}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Target data shape: {y_train.shape}")
    print(f"Data types: {X_train.dtypes}")
    
    for model_name, model in models.items():
        param_grid = param_grids.get(model_name, {})

        if not param_grid:
            print(f"⚠️ Warning: No parameters found for {model_name}, skipping GridSearchCV.")
            continue  # Skip models without hyperparameter grids

        try:
            print(f"\nTraining {model_name} with params {param_grid}")
            # Add data validation
            if X_train.isnull().any().any():
                print(f"⚠️ Warning: Found NaN values in features for {model_name}")
            if y_train.isnull().any():
                print(f"⚠️ Warning: Found NaN values in target for {model_name}")
            
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2' if model_type == 'regression' else 'accuracy')
            grid_search.fit(X_train, y_train)
            score = grid_search.best_score_
            print(f"✅ {model_name} Best Score: {score}")
            print(f"Best parameters: {grid_search.best_params_}")
            
            if score > best_score:
                best_model = grid_search.best_estimator_
                best_score = score
                print(f"🎯 New best model found: {model_name} with score {score}")

        except Exception as e:
            print(f"❌ Error training {model_name}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            continue  # Skip this model and move to the next one

    if best_model is None:
        print("\n❌ Auto-training failed: No valid model found.")
        print("Attempting fallback to simple model...")
        try:
            if model_type == 'regression':
                print("Trying LinearRegression as fallback")
                fallback_model = LinearRegression()
            else:
                print("Trying LogisticRegression as fallback")
                fallback_model = LogisticRegression()
            
            # Add data preprocessing for fallback
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            fallback_model.fit(X_train_scaled, y_train)
            best_model = fallback_model
            print("✅ Fallback model trained successfully")
        except Exception as e:
            print(f"❌ Fallback model also failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Traceback: {traceback.format_exc()}")
    else:
        print(f"\n🏆 Best Model Selected: {best_model} with Score: {best_score}")

    return best_model

MODEL_FOLDER = os.path.join(os.path.dirname(__file__), "models")


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    filename = secure_filename(filename)
    filepath = os.path.join(MODEL_FOLDER, filename)
    print(f"[DEBUG] Looking for: {filepath}")  # For debug

    if not os.path.exists(filepath):
        print("[DEBUG] File not found.")
        return jsonify({"error": "File not found"}), 404

    return send_from_directory(MODEL_FOLDER, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)