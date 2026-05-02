import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingAgent:
    csv_path: str
    target_column: str
    test_size: float = 0.2
    random_state: int = 42
    numeric_impute_strategy: str = 'mean'
    categorical_impute_strategy: str = 'most_frequent'
    scale_numeric: bool = True
    encode_categorical: bool = True

    data: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        logger.info('Loading dataset from %s', self.csv_path)
        self.data = pd.read_csv(self.csv_path)
        logger.info('Loaded dataset shape: %s', self.data.shape)
        return self.data

    def clean_data(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError('Data not loaded. Call load_data() first.')

        logger.info('Cleaning data: handling missing values and basic duplicates')

        self.data = self.data.drop_duplicates().reset_index(drop=True)

        # Numeric missing values
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            if self.data[col].isna().any():
                fill = self.data[col].mean() if self.numeric_impute_strategy == 'mean' else self.data[col].median()
                self.data[col] = self.data[col].fillna(fill)
                logger.debug('Imputed missing numeric %s with %s', col, fill)

        # Categorical missing values
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in categorical_cols:
            if self.data[col].isna().any():
                fill = self.data[col].mode().iloc[0]
                self.data[col] = self.data[col].fillna(fill)
                logger.debug('Imputed missing categorical %s with %s', col, fill)

        # Handle infinities and remaining nulls after imputation
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data[numeric_cols] = self.data[numeric_cols].fillna(0)

        return self.data

    def feature_engineering(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError('Data not loaded. Call load_data() first.')

        logger.info('Performing basic feature engineering')

        # Example feature engineering for credit risk data
        if 'LIMIT_BAL' in self.data.columns and 'BILL_AMT1' in self.data.columns:
            pbill = self.data['BILL_AMT1'].replace(0, np.nan)
            self.data['limit_to_bill_ratio'] = self.data['LIMIT_BAL'] / pbill

        if 'PAY_AMT1' in self.data.columns and 'BILL_AMT1' in self.data.columns:
            pbill = self.data['BILL_AMT1'].replace(0, np.nan)
            self.data['pay_to_bill_ratio'] = self.data['PAY_AMT1'] / pbill

        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data['limit_to_bill_ratio'] = self.data['limit_to_bill_ratio'].fillna(0)
        self.data['pay_to_bill_ratio'] = self.data['pay_to_bill_ratio'].fillna(0)

        logger.debug('Engineered features: %s', [c for c in self.data.columns if 'ratio' in c])
        return self.data

    def split_data(self) -> Dict[str, pd.DataFrame]:
        if self.data is None:
            raise ValueError('Data not loaded. Call load_data() first.')

        logger.info('Splitting data into train/test')

        if self.target_column not in self.data.columns:
            raise ValueError('Target column %s not found in data' % self.target_column)

        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        transformers = []
        if self.encode_categorical and categorical_features:
            transformers.append(
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
            )

        if self.scale_numeric and numeric_features:
            transformers.append(
                ('num', StandardScaler(), numeric_features)
            )

        preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')

        X_processed = preprocessor.fit_transform(X)

        # keep columns names minimal
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        logger.info('Split sizes: X_train=%s, X_test=%s', X_train.shape, X_test.shape)

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
        }

    def process(self) -> Dict[str, pd.DataFrame]:
        self.load_data()
        self.clean_data()
        self.feature_engineering()
        result = self.split_data()
        logger.info('Process complete')
        return result
