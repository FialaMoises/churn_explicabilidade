"""
Data Preprocessor Module
Responsável por preprocessamento e limpeza de dados
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Classe para preprocessamento de dados"""

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = None

    def handle_missing_values(self, df: pd.DataFrame,
                             strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Trata valores missing

        Args:
            df: DataFrame original
            strategy: Dicionário com estratégia por tipo {'numerical': 'median', 'categorical': 'most_frequent'}

        Returns:
            DataFrame com valores tratados
        """
        if strategy is None:
            strategy = {
                'numerical': 'median',
                'categorical': 'most_frequent'
            }

        df_clean = df.copy()

        numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns

        if len(numerical_cols) > 0 and df_clean[numerical_cols].isnull().any().any():
            imputer_num = SimpleImputer(strategy=strategy['numerical'])
            df_clean[numerical_cols] = imputer_num.fit_transform(df_clean[numerical_cols])
            self.imputers['numerical'] = imputer_num
            logger.info(f"Missing values tratados em colunas numéricas usando {strategy['numerical']}")

        if len(categorical_cols) > 0 and df_clean[categorical_cols].isnull().any().any():
            imputer_cat = SimpleImputer(strategy=strategy['categorical'])
            df_clean[categorical_cols] = imputer_cat.fit_transform(df_clean[categorical_cols])
            self.imputers['categorical'] = imputer_cat
            logger.info(f"Missing values tratados em colunas categóricas usando {strategy['categorical']}")

        return df_clean

    def encode_categorical(self, df: pd.DataFrame,
                          categorical_cols: List[str] = None,
                          method: str = 'label') -> pd.DataFrame:
        """
        Codifica variáveis categóricas

        Args:
            df: DataFrame original
            categorical_cols: Lista de colunas categóricas
            method: 'label' para LabelEncoder ou 'onehot' para One-Hot Encoding

        Returns:
            DataFrame com variáveis codificadas
        """
        df_encoded = df.copy()

        if categorical_cols is None:
            categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()

        if method == 'label':
            for col in categorical_cols:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
                logger.info(f"Coluna '{col}' codificada usando LabelEncoder")

        elif method == 'onehot':
            df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, prefix=categorical_cols)
            logger.info(f"Colunas {categorical_cols} codificadas usando One-Hot Encoding")

        return df_encoded

    def scale_features(self, df: pd.DataFrame,
                      numerical_cols: List[str] = None,
                      method: str = 'standard') -> pd.DataFrame:
        """
        Normaliza/padroniza features numéricas

        Args:
            df: DataFrame original
            numerical_cols: Lista de colunas numéricas
            method: 'standard' para StandardScaler

        Returns:
            DataFrame com features escaladas
        """
        df_scaled = df.copy()

        if numerical_cols is None:
            numerical_cols = df_scaled.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if method == 'standard':
            scaler = StandardScaler()
            df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
            self.scalers['standard'] = scaler
            logger.info(f"Features numéricas padronizadas: {len(numerical_cols)} colunas")

        return df_scaled

    def remove_outliers(self, df: pd.DataFrame,
                       columns: List[str] = None,
                       method: str = 'iqr',
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers

        Args:
            df: DataFrame original
            columns: Colunas para verificar outliers
            method: 'iqr' ou 'zscore'
            threshold: Threshold para remoção (IQR multiplier ou z-score)

        Returns:
            DataFrame sem outliers
        """
        df_clean = df.copy()

        if columns is None:
            columns = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()

        initial_rows = len(df_clean)

        for col in columns:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean = df_clean[z_scores < threshold]

        removed_rows = initial_rows - len(df_clean)
        logger.info(f"Outliers removidos: {removed_rows} linhas ({removed_rows/initial_rows*100:.2f}%)")

        return df_clean

    def create_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features binárias adicionais

        Args:
            df: DataFrame original

        Returns:
            DataFrame com features binárias adicionais
        """
        df_new = df.copy()

        if 'Balance' in df_new.columns:
            df_new['Has_Zero_Balance'] = (df_new['Balance'] == 0).astype(int)

        if 'Age' in df_new.columns:
            df_new['Is_Senior'] = (df_new['Age'] >= 60).astype(int)

        if 'NumOfProducts' in df_new.columns:
            df_new['Has_Multiple_Products'] = (df_new['NumOfProducts'] > 1).astype(int)

        logger.info(f"Features binárias criadas: {len([c for c in df_new.columns if c not in df.columns])}")

        return df_new

    def preprocess_pipeline(self, df: pd.DataFrame,
                           fit: bool = True,
                           remove_outliers_flag: bool = False) -> pd.DataFrame:
        """
        Pipeline completo de preprocessamento

        Args:
            df: DataFrame original
            fit: Se True, ajusta os transformers. Se False, apenas transforma
            remove_outliers_flag: Se True, remove outliers

        Returns:
            DataFrame preprocessado
        """
        logger.info("Iniciando pipeline de preprocessamento...")

        df_processed = df.copy()

        df_processed = self.handle_missing_values(df_processed)

        df_processed = self.create_binary_features(df_processed)

        categorical_cols = ['Geography', 'Gender', 'Card Type']
        categorical_cols = [c for c in categorical_cols if c in df_processed.columns]
        if categorical_cols:
            df_processed = self.encode_categorical(df_processed, categorical_cols, method='label')

        if remove_outliers_flag and fit:
            df_processed = self.remove_outliers(df_processed, method='iqr', threshold=3.0)

        if fit:
            self.feature_names = df_processed.columns.tolist()

        logger.info("Pipeline de preprocessamento concluído!")

        return df_processed


if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader()
    df = loader.load_data()
    X, y = loader.split_features_target()

    preprocessor = DataPreprocessor()
    X_processed = preprocessor.preprocess_pipeline(X, fit=True)

    print("\n=== Preprocessamento Concluído ===")
    print(f"Shape original: {X.shape}")
    print(f"Shape processado: {X_processed.shape}")
    print(f"\nPrimeiras linhas:\n{X_processed.head()}")
    print(f"\nInfo:\n{X_processed.info()}")
