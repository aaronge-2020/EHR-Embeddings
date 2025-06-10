"""
Machine Learning Pipeline for EHR Embeddings
"""
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
# Optional imports with fallbacks
HAS_XGBOOST = False
HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    pass

from config import Config

# Configure logging
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class EHRMLPipeline:
    """
    Machine Learning Pipeline for EHR Embeddings
    """
    
    def __init__(self, task_type: str = "classification"):
        """
        Initialize the ML pipeline
        
        Args:
            task_type: Either 'classification' or 'regression'
        """
        self.task_type = task_type.lower()
        if self.task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be 'classification' or 'regression'")
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        
        logger.info(f"Initialized EHRMLPipeline for {self.task_type}")
    
    def _get_models(self) -> Dict[str, Any]:
        """Get available models based on task type"""
        models = {}
        
        if self.task_type == "classification":
            models.update({
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
                "svm": SVC(random_state=42, probability=True),
            })
            
            if HAS_XGBOOST:
                models["xgboost"] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
            
            if HAS_LIGHTGBM:
                models["lightgbm"] = lgb.LGBMClassifier(random_state=42, verbose=-1)
                
        else:  # regression
            models.update({
                "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "linear_regression": LinearRegression(),
                "svm": SVR(),
            })
            
            if HAS_XGBOOST:
                models["xgboost"] = xgb.XGBRegressor(random_state=42)
            
            if HAS_LIGHTGBM:
                models["lightgbm"] = lgb.LGBMRegressor(random_state=42, verbose=-1)
        
        return models
    
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    target_column: str,
                    embedding_columns: Optional[List[str]] = None,
                    additional_features: Optional[List[str]] = None,
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with embeddings and target
            target_column: Name of the target column
            embedding_columns: List of embedding column names (if None, will auto-detect)
            additional_features: List of additional feature columns to include
            test_size: Test set size
            random_state: Random state for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Preparing data for training")
        
        # Auto-detect embedding columns if not provided
        if embedding_columns is None:
            embedding_columns = [col for col in df.columns if 'embedding' in col.lower()]
        
        # Handle embeddings stored as arrays in single column
        if len(embedding_columns) == 1 and hasattr(df[embedding_columns[0]].iloc[0], '__len__'):
            # Embeddings are stored as arrays in a single column
            embeddings_data = np.stack(df[embedding_columns[0]].values)
        else:
            # Embeddings are stored as separate columns
            embeddings_data = df[embedding_columns].values
        
        # Add additional features if specified
        if additional_features:
            additional_data = df[additional_features].values
            X = np.concatenate([embeddings_data, additional_data], axis=1)
        else:
            X = embeddings_data
        
        # Prepare target variable
        y = df[target_column].values
        
        # Encode categorical targets for classification
        if self.task_type == "classification" and y.dtype == 'object':
            self.encoders['target'] = LabelEncoder()
            y = self.encoders['target'].fit_transform(y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if self.task_type == "classification" else None
        )
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_train = self.scalers['features'].fit_transform(X_train)
        X_test = self.scalers['features'].transform(X_test)
        
        logger.info(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        logger.info(f"Feature dimensions: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self, 
                    X_train: np.ndarray, 
                    y_train: np.ndarray,
                    cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """
        Train multiple models and evaluate with cross-validation
        
        Args:
            X_train: Training features
            y_train: Training targets
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with model performance scores
        """
        logger.info("Training multiple models")
        
        models = self._get_models()
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}")
            
            try:
                # Cross-validation
                if self.task_type == "classification":
                    cv_scores = cross_val_score(model, X_train, y_train, 
                                              cv=cv_folds, scoring='accuracy')
                    metric_name = "accuracy"
                else:
                    cv_scores = cross_val_score(model, X_train, y_train, 
                                              cv=cv_folds, scoring='neg_mean_squared_error')
                    cv_scores = -cv_scores  # Convert to positive MSE
                    metric_name = "mse"
                
                # Train on full training set
                model.fit(X_train, y_train)
                self.models[name] = model
                
                # Store results
                results[name] = {
                    f"cv_{metric_name}_mean": cv_scores.mean(),
                    f"cv_{metric_name}_std": cv_scores.std()
                }
                
                logger.info(f"{name} - CV {metric_name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                results[name] = {"error": str(e)}
        
        # Find best model
        if self.task_type == "classification":
            best_name = max([k for k in results.keys() if "error" not in results[k]], 
                          key=lambda x: results[x]["cv_accuracy_mean"])
        else:
            best_name = min([k for k in results.keys() if "error" not in results[k]], 
                          key=lambda x: results[x]["cv_mse_mean"])
        
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        logger.info(f"Best model: {best_name}")
        
        return results
    
    def evaluate_model(self, 
                      model: Any, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        if self.task_type == "classification":
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1": f1_score(y_test, y_pred, average='weighted')
            }
        else:
            metrics = {
                "mse": mean_squared_error(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
            }
        
        return metrics
    
    def hyperparameter_tuning(self, 
                            X_train: np.ndarray, 
                            y_train: np.ndarray,
                            model_name: str = "random_forest",
                            param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model
        
        Args:
            X_train: Training features
            y_train: Training targets
            model_name: Name of the model to tune
            param_grid: Parameter grid for tuning
            
        Returns:
            Best parameters and score
        """
        logger.info(f"Hyperparameter tuning for {model_name}")
        
        models = self._get_models()
        if model_name not in models:
            raise ValueError(f"Model {model_name} not available")
        
        model = models[model_name]
        
        # Default parameter grids
        default_param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2]
            },
            "lightgbm": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 6, 9],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        }
        
        if param_grid is None:
            param_grid = default_param_grids.get(model_name, {})
        
        if not param_grid:
            logger.warning(f"No parameter grid provided for {model_name}")
            return {}
        
        # Perform grid search
        scoring = 'accuracy' if self.task_type == "classification" else 'neg_mean_squared_error'
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "best_model": grid_search.best_estimator_
        }
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return results
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Args:
            model_name: Name of the model (if None, uses best model)
            
        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            model = self.best_model
            model_name = self.best_model_name
        else:
            model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).flatten()
        else:
            logger.warning(f"Model {model_name} does not have feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        feature_names = [f"feature_{i}" for i in range(len(importance))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        return importance_df
    
    def save_model(self, model_name: str, filepath: Optional[Path] = None):
        """
        Save a trained model
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model (if None, uses default)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        if filepath is None:
            filepath = Config.MODEL_OUTPUT_DIR / f"{model_name}_model.joblib"
        
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: Path):
        """
        Load a trained model
        
        Args:
            model_name: Name to assign to the loaded model
            filepath: Path to the model file
        """
        self.models[model_name] = joblib.load(filepath)
        logger.info(f"Model {model_name} loaded from {filepath}")
    
    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """
        Make predictions using a trained model
        
        Args:
            X: Features to predict on
            model_name: Name of the model to use (if None, uses best model)
            
        Returns:
            Predictions
        """
        if model_name is None:
            model = self.best_model
        else:
            model = self.models.get(model_name)
        
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        # Scale features if scaler is available
        if 'features' in self.scalers:
            X = self.scalers['features'].transform(X)
        
        return model.predict(X)


def create_sample_ehr_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample EHR data for testing
    
    Args:
        n_samples: Number of samples to create
        
    Returns:
        DataFrame with sample EHR data
    """
    np.random.seed(42)
    
    # Sample medical conditions and symptoms
    conditions = ["diabetes", "hypertension", "asthma", "pneumonia", "bronchitis"]
    symptoms = ["fever", "cough", "fatigue", "shortness of breath", "chest pain"]
    
    data = []
    for i in range(n_samples):
        # Generate random text data
        chief_complaint = np.random.choice(symptoms)
        diagnosis = np.random.choice(conditions)
        
        # Create embeddings (simulated)
        embedding = np.random.normal(0, 1, 768)  # Simulating Gemini embedding size
        
        # Create binary target (e.g., readmission risk)
        target = np.random.choice([0, 1], p=[0.7, 0.3])
        
        data.append({
            'patient_id': f"P{i:04d}",
            'chief_complaint': chief_complaint,
            'diagnosis': diagnosis,
            'embedding': embedding,
            'readmission_risk': target
        })
    
    return pd.DataFrame(data)