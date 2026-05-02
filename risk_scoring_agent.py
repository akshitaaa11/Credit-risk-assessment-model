import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class RiskScoringAgent:
    data: Dict[str, Any]
    test_size: float = 0.2
    random_state: int = 42
    smote_ratio: float = 1.0
    model_dir: str = 'models'
    models: Dict[str, Any] = field(default_factory=dict)
    trained_models: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    best_model_name: Optional[str] = None

    def __post_init__(self):
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)

    def apply_smote(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, np.ndarray]:
        logger.info('Applying SMOTE on training data for class imbalance handling')
        smote = SMOTE(sampling_strategy=self.smote_ratio, random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info('After SMOTE: %s', np.bincount(y_res))
        return {'X_res': X_res, 'y_res': y_res}

    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        logger.info('Training models: LogisticRegression, RandomForest, XGBoost')

        self.models = {
            'logistic_regression': Pipeline([
                ("scaler", RobustScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=self.random_state))
            ]),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=self.random_state),
        }

        for name, model in self.models.items():
            logger.info('Training %s', name)
            model.fit(X_train, y_train)
            self.trained_models[name] = model
            self._save_model(name, model)
            logger.info('%s training complete', name)

        return self.trained_models

    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        logger.info('Evaluating trained models')
        self.metrics = {}

        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)

            self.metrics[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_prob),
            }

            logger.info('%s metrics: %s', name, self.metrics[name])

        return self.metrics

    def select_best_model(self, metric: str = 'roc_auc') -> Dict[str, Any]:
        if not self.metrics:
            raise ValueError('No metrics to select best model from. Run evaluate_models first.')

        self.best_model_name = max(self.metrics, key=lambda k: self.metrics[k].get(metric, float('-inf')))
        best_model = self.trained_models[self.best_model_name]

        logger.info('Selected best model: %s based on %s', self.best_model_name, metric)

        return {'best_model_name': self.best_model_name, 'best_model': best_model}

    def predict(self, X_new: np.ndarray) -> Dict[str, Any]:
        if self.best_model_name is None:
            raise ValueError('Best model not selected yet. Call select_best_model() first.')

        best_model = self.trained_models[self.best_model_name]

        preds = best_model.predict(X_new)
        pd_values = best_model.predict_proba(X_new)[:, 1] if hasattr(best_model, 'predict_proba') else None

        logger.info('Predictions generated with %s', self.best_model_name)

        return {'labels': preds.tolist(), 'pd': pd_values.tolist() if pd_values is not None else None}

    def _save_model(self, name: str, model: Any) -> None:
        path = Path(self.model_dir) / f'{name}.joblib'
        joblib.dump(model, path)
        logger.info('Saved model %s to %s', name, path)

    def run(self) -> Dict[str, Any]:
        logger.info('Running full risk scoring workflow with pre-split data')

        X_train = self.data.get('X_train')
        X_test = self.data.get('X_test')
        y_train = self.data.get('y_train')
        y_test = self.data.get('y_test')

        if X_train is None or X_test is None or y_train is None or y_test is None:
            raise ValueError('Data must include X_train, X_test, y_train, y_test')

        smote_res = self.apply_smote(X_train, y_train)
        X_res = smote_res['X_res']
        y_res = smote_res['y_res']

        self.train_models(X_res, y_res)
        self.evaluate_models(X_test, y_test)
        best = self.select_best_model()
        predictions = self.predict(X_test)

        return {
            'metrics': self.metrics,
            'best_model': best,
            'predictions': predictions,
        }
