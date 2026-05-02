"""
Inference module for Credit Risk Assessment System.

This module loads trained model artifacts and provides prediction functionality
for LangGraph agents. Supports single and batch predictions.
"""

import os
import json
import pickle
import warnings
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class CreditRiskPredictor:
    """
    Inference engine for credit risk prediction.
    
    Loads trained model artifacts and provides probability of default (PD)
    predictions with risk classification.
    """
    
    # Risk thresholds (can be adjusted based on business requirements)
    RISK_THRESHOLDS = {
        'low': 0.25,        # PD < 25%
        'medium': 0.50,     # 25% <= PD < 50%
        'high': 0.75,       # 50% <= PD < 75%
        'critical': 1.00    # PD >= 75%
    }
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize predictor and load model artifacts.
        
        Parameters:
            model_dir: Directory containing saved model artifacts
        
        Raises:
            FileNotFoundError: If required artifacts are missing
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_names = None
        self.threshold = 0.5
        
        self._load_artifacts()
    
    def _load_artifacts(self) -> None:
        """Load model, scaler, and metadata from disk."""
        # Load model
        model_path = os.path.join(self.model_dir, "best_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        scaler_path = os.path.join(self.model_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load metadata
        metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.feature_names = self.metadata.get('feature_names', [])
                self.threshold = self.metadata.get('threshold', 0.5)
        
        print(f"✓ Model loaded: {self.metadata.get('model_name', 'unknown')}")
        print(f"✓ Optimal threshold: {self.threshold:.4f}")
        print(f"✓ Expected features: {len(self.feature_names)}")
    
    def _classify_risk(self, pd_score: float) -> str:
        """
        Classify risk level based on PD score.
        
        Parameters:
            pd_score: Probability of default (0.0 to 1.0)
        
        Returns:
            Risk class: 'Low', 'Medium', 'High', or 'Critical'
        """
        if pd_score < self.RISK_THRESHOLDS['low']:
            return "Low"
        elif pd_score < self.RISK_THRESHOLDS['medium']:
            return "Medium"
        elif pd_score < self.RISK_THRESHOLDS['high']:
            return "High"
        else:
            return "Critical"
    
    def _calculate_confidence(self, pd_score: float) -> Dict[str, Any]:
        """
        Calculate prediction confidence metadata.
        
        Parameters:
            pd_score: Probability of default
        
        Returns:
            Dictionary with confidence metrics
        """
        # Distance from decision threshold
        distance_from_threshold = abs(pd_score - self.threshold)
        
        # Confidence score (higher when further from threshold)
        # Normalized to 0-1 scale
        confidence_score = min(distance_from_threshold * 2, 1.0)
        
        # Confidence level
        if confidence_score > 0.7:
            confidence_level = "High"
        elif confidence_score > 0.4:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"
        
        return {
            'confidence_score': round(confidence_score, 4),
            'confidence_level': confidence_level,
            'distance_from_threshold': round(distance_from_threshold, 4),
            'threshold_used': self.threshold
        }
    
    def _validate_features(self, features: Union[Dict, pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Validate and prepare features for prediction.
        
        Parameters:
            features: Input features (dict, DataFrame, or array)
        
        Returns:
            Validated DataFrame
        
        Raises:
            ValueError: If features are invalid or incomplete
        """
        # Convert to DataFrame if needed
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        elif isinstance(features, np.ndarray):
            if len(self.feature_names) == 0:
                raise ValueError("Feature names not available. Cannot validate array input.")
            df = pd.DataFrame(features, columns=self.feature_names)
        elif isinstance(features, pd.DataFrame):
            df = features.copy()
        else:
            raise ValueError(f"Unsupported feature type: {type(features)}")
        
        # Check for missing features
        if len(self.feature_names) > 0:
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Reorder columns to match training
            df = df[self.feature_names]
        
        return df
    
    def predict(
        self, 
        features: Union[Dict, pd.DataFrame, np.ndarray],
        include_confidence: bool = True,
        use_optimal_threshold: bool = True
    ) -> Dict[str, Any]:
        """
        Predict probability of default for a single borrower.
        
        Parameters:
            features: Borrower features (dict, DataFrame, or array)
            include_confidence: Whether to include confidence metrics
            use_optimal_threshold: Whether to use optimized threshold for classification
        
        Returns:
            Dictionary containing:
                - pd_score: Probability of default (0.0 to 1.0)
                - risk_class: Risk classification ('Low', 'Medium', 'High', 'Critical')
                - prediction: Binary prediction (0=no default, 1=default)
                - confidence: Confidence metadata (if include_confidence=True)
        
        Example:
            >>> predictor = CreditRiskPredictor()
            >>> features = {'age': 35, 'income': 50000, ...}
            >>> result = predictor.predict(features)
            >>> print(result['pd_score'], result['risk_class'])
        """
        # Validate features
        df = self._validate_features(features)
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        # Predict probability
        pd_score = self.model.predict_proba(X_scaled)[0, 1]
        
        # Classify risk
        risk_class = self._classify_risk(pd_score)
        
        # Binary prediction
        threshold = self.threshold if use_optimal_threshold else 0.5
        prediction = 1 if pd_score >= threshold else 0
        
        # Build result
        result = {
            'pd_score': round(float(pd_score), 4),
            'risk_class': risk_class,
            'prediction': int(prediction)
        }
        
        # Add confidence if requested
        if include_confidence:
            result['confidence'] = self._calculate_confidence(pd_score)
        
        return result
    
    def predict_batch(
        self, 
        features_list: Union[List[Dict], pd.DataFrame],
        include_confidence: bool = True,
        use_optimal_threshold: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict probability of default for multiple borrowers.
        
        Parameters:
            features_list: List of feature dicts or DataFrame with multiple rows
            include_confidence: Whether to include confidence metrics
            use_optimal_threshold: Whether to use optimized threshold
        
        Returns:
            List of prediction dictionaries
        
        Example:
            >>> predictor = CreditRiskPredictor()
            >>> borrowers = [
            ...     {'age': 35, 'income': 50000, ...},
            ...     {'age': 42, 'income': 75000, ...}
            ... ]
            >>> results = predictor.predict_batch(borrowers)
        """
        # Convert to DataFrame
        if isinstance(features_list, list):
            df = pd.DataFrame(features_list)
        elif isinstance(features_list, pd.DataFrame):
            df = features_list.copy()
        else:
            raise ValueError("features_list must be a list of dicts or DataFrame")
        
        # Validate features
        df = self._validate_features(df)
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        # Predict probabilities
        pd_scores = self.model.predict_proba(X_scaled)[:, 1]
        
        # Generate predictions for each borrower
        results = []
        threshold = self.threshold if use_optimal_threshold else 0.5
        
        for pd_score in pd_scores:
            risk_class = self._classify_risk(pd_score)
            prediction = 1 if pd_score >= threshold else 0
            
            result = {
                'pd_score': round(float(pd_score), 4),
                'risk_class': risk_class,
                'prediction': int(prediction)
            }
            
            if include_confidence:
                result['confidence'] = self._calculate_confidence(pd_score)
            
            results.append(result)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        if self.metadata is None:
            return {
                'model_loaded': True,
                'model_type': str(type(self.model).__name__),
                'threshold': self.threshold
            }
        
        return {
            'model_name': self.metadata.get('model_name'),
            'model_type': str(type(self.model).__name__),
            'threshold': self.threshold,
            'training_date': self.metadata.get('training_date'),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'performance_metrics': self.metadata.get('results', {})
        }
    
    def update_risk_thresholds(
        self, 
        low: float = None, 
        medium: float = None, 
        high: float = None
    ) -> None:
        """
        Update risk classification thresholds.
        
        Parameters:
            low: Threshold for Low risk (default: 0.25)
            medium: Threshold for Medium risk (default: 0.50)
            high: Threshold for High risk (default: 0.75)
        
        Example:
            >>> predictor.update_risk_thresholds(low=0.20, medium=0.45, high=0.70)
        """
        if low is not None:
            self.RISK_THRESHOLDS['low'] = low
        if medium is not None:
            self.RISK_THRESHOLDS['medium'] = medium
        if high is not None:
            self.RISK_THRESHOLDS['high'] = high
        
        print("✓ Risk thresholds updated:")
        print(f"  Low:      < {self.RISK_THRESHOLDS['low']:.2%}")
        print(f"  Medium:   < {self.RISK_THRESHOLDS['medium']:.2%}")
        print(f"  High:     < {self.RISK_THRESHOLDS['high']:.2%}")
        print(f"  Critical: >= {self.RISK_THRESHOLDS['high']:.2%}")


# Convenience functions for LangGraph agents
def load_predictor(model_dir: str = "models") -> CreditRiskPredictor:
    """
    Load predictor instance.
    
    Parameters:
        model_dir: Directory containing model artifacts
    
    Returns:
        Initialized CreditRiskPredictor
    
    Example:
        >>> predictor = load_predictor("models")
    """
    return CreditRiskPredictor(model_dir=model_dir)


def predict_default_probability(
    features: Union[Dict, pd.DataFrame],
    model_dir: str = "models",
    include_confidence: bool = True
) -> Dict[str, Any]:
    """
    Quick prediction function for single borrower.
    
    Parameters:
        features: Borrower features
        model_dir: Directory containing model artifacts
        include_confidence: Whether to include confidence metrics
    
    Returns:
        Prediction dictionary
    
    Example:
        >>> result = predict_default_probability({'age': 35, 'income': 50000, ...})
        >>> print(result['pd_score'], result['risk_class'])
    """
    predictor = CreditRiskPredictor(model_dir=model_dir)
    return predictor.predict(features, include_confidence=include_confidence)


def predict_batch_default_probability(
    features_list: Union[List[Dict], pd.DataFrame],
    model_dir: str = "models",
    include_confidence: bool = True
) -> List[Dict[str, Any]]:
    """
    Quick batch prediction function.
    
    Parameters:
        features_list: List of borrower features or DataFrame
        model_dir: Directory containing model artifacts
        include_confidence: Whether to include confidence metrics
    
    Returns:
        List of prediction dictionaries
    
    Example:
        >>> borrowers = [{'age': 35, ...}, {'age': 42, ...}]
        >>> results = predict_batch_default_probability(borrowers)
    """
    predictor = CreditRiskPredictor(model_dir=model_dir)
    return predictor.predict_batch(features_list, include_confidence=include_confidence)


if __name__ == "__main__":
    """
    Example usage and testing.
    """
    print("="*60)
    print("CREDIT RISK PREDICTOR - INFERENCE MODULE")
    print("="*60)
    
    try:
        # Initialize predictor
        predictor = CreditRiskPredictor(model_dir="models")
        
        # Display model info
        print("\nModel Information:")
        info = predictor.get_model_info()
        for key, value in info.items():
            if key not in ['feature_names', 'performance_metrics']:
                print(f"  {key}: {value}")
        
        # Example prediction (requires actual features)
        print("\nExample usage:")
        print(">>> features = {'age': 35, 'income': 50000, ...}")
        print(">>> result = predictor.predict(features)")
        print(">>> print(result)")
        print("\nPredictor ready for inference!")
        
    except FileNotFoundError as e:
        print(f"\n⚠ {e}")
        print("\nPlease run train.py first to generate model artifacts.")
    except Exception as e:
        print(f"\n⚠ Error: {e}")