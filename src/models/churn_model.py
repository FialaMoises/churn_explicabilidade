"""
Churn Model Module
Classe principal para treinamento e predição de churn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, precision_recall_curve, f1_score,
                            classification_report, confusion_matrix, brier_score_loss,
                            precision_score, recall_score, accuracy_score)
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Tuple, Any
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModel:
    """Classe principal para modelagem de churn"""

    def __init__(self, model_type: str = 'xgboost', config: Dict = None):
        """
        Inicializa o modelo de churn

        Args:
            model_type: Tipo de modelo ('logistic', 'random_forest', 'xgboost', 'lightgbm')
            config: Dicionário com configurações do modelo
        """
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.feature_names = None
        self.is_fitted = False
        self.metrics = {}
        self.best_threshold = 0.5

        self._initialize_model()

    def _initialize_model(self):
        """Inicializa o modelo baseado no tipo especificado"""

        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                C=self.config.get('C', 1.0),
                max_iter=self.config.get('max_iter', 1000),
                class_weight=self.config.get('class_weight', 'balanced'),
                random_state=self.config.get('random_state', 42)
            )

        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 200),
                max_depth=self.config.get('max_depth', 10),
                min_samples_split=self.config.get('min_samples_split', 20),
                min_samples_leaf=self.config.get('min_samples_leaf', 10),
                class_weight=self.config.get('class_weight', 'balanced'),
                random_state=self.config.get('random_state', 42),
                n_jobs=self.config.get('n_jobs', -1)
            )

        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=self.config.get('n_estimators', 200),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                scale_pos_weight=self.config.get('scale_pos_weight', 3),
                random_state=self.config.get('random_state', 42),
                tree_method=self.config.get('tree_method', 'hist'),
                enable_categorical=False
            )

        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=self.config.get('n_estimators', 200),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                subsample=self.config.get('subsample', 0.8),
                colsample_bytree=self.config.get('colsample_bytree', 0.8),
                is_unbalance=self.config.get('is_unbalance', True),
                random_state=self.config.get('random_state', 42),
                verbose=self.config.get('verbose', -1)
            )

        else:
            raise ValueError(f"Tipo de modelo não suportado: {self.model_type}")

        logger.info(f"Modelo {self.model_type} inicializado")

    def train(self, X: pd.DataFrame, y: pd.Series,
             validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Treina o modelo

        Args:
            X: Features de treino
            y: Target de treino
            validation_split: Proporção dos dados para validação

        Returns:
            Dicionário com métricas de treino
        """
        logger.info(f"Iniciando treino do modelo {self.model_type}...")

        X_numeric = X.select_dtypes(include=[np.number])

        X_train, X_val, y_train, y_val = train_test_split(
            X_numeric, y,
            test_size=validation_split,
            stratify=y,
            random_state=42
        )

        logger.info(f"Treino: {X_train.shape[0]} amostras, Validação: {X_val.shape[0]} amostras")

        self.model.fit(X_train, y_train)
        self.feature_names = X_numeric.columns.tolist()
        self.is_fitted = True

        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        self.metrics = {
            'train': self._calculate_metrics(X_train, y_train),
            'validation': self._calculate_metrics(X_val, y_val)
        }

        logger.info("Treino concluído!")
        logger.info(f"ROC-AUC (validação): {self.metrics['validation']['roc_auc']:.4f}")
        logger.info(f"F1-Score (validação): {self.metrics['validation']['f1_score']:.4f}")

        return self.metrics

    def _calculate_metrics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calcula métricas de avaliação

        Args:
            X: Features
            y: Target real

        Returns:
            Dicionário com métricas
        """
        X_numeric = X.select_dtypes(include=[np.number])
        y_pred_proba = self.model.predict_proba(X_numeric)[:, 1]
        y_pred = (y_pred_proba >= self.best_threshold).astype(int)

        metrics = {
            'roc_auc': roc_auc_score(y, y_pred_proba),
            'f1_score': f1_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'accuracy': accuracy_score(y, y_pred),
            'brier_score': brier_score_loss(y, y_pred_proba)
        }

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prediz probabilidades de churn

        Args:
            X: Features

        Returns:
            Array com probabilidades
        """
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")

        X_numeric = X.select_dtypes(include=[np.number])
        return self.model.predict_proba(X_numeric)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """
        Prediz churn (0 ou 1)

        Args:
            X: Features
            threshold: Threshold de decisão (usa best_threshold se None)

        Returns:
            Array com predições
        """
        if threshold is None:
            threshold = self.best_threshold

        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                      cv_folds: int = 5) -> Dict[str, Any]:
        """
        Validação cruzada

        Args:
            X: Features
            y: Target
            cv_folds: Número de folds

        Returns:
            Dicionário com resultados da CV
        """
        logger.info(f"Executando validação cruzada com {cv_folds} folds...")

        X_numeric = X.select_dtypes(include=[np.number])
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        cv_scores = cross_val_score(self.model, X_numeric, y, cv=skf, scoring='roc_auc')

        cv_results = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }

        logger.info(f"ROC-AUC (CV): {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")

        return cv_results

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Retorna feature importance

        Args:
            top_n: Número de features mais importantes

        Returns:
            DataFrame com importâncias
        """
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            raise ValueError("Modelo não suporta feature importance")

        feature_imp = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)

        return feature_imp

    def optimize_threshold(self, X: pd.DataFrame, y: pd.Series,
                          cost_matrix: Dict = None) -> float:
        """
        Otimiza threshold baseado em custo ou F1

        Args:
            X: Features de validação
            y: Target de validação
            cost_matrix: Dicionário com custos (opcional)

        Returns:
            Threshold ótimo
        """
        logger.info("Otimizando threshold...")

        X_numeric = X.select_dtypes(include=[np.number])
        y_pred_proba = self.model.predict_proba(X_numeric)[:, 1]

        if cost_matrix is None:
            precision, recall, thresholds = precision_recall_curve(y, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            self.best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            logger.info(f"Threshold ótimo (F1): {self.best_threshold:.3f}")
        else:
            self.best_threshold = 0.5

        return self.best_threshold

    def save_model(self, filepath: str):
        """
        Salva o modelo treinado

        Args:
            filepath: Caminho para salvar o modelo
        """
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'best_threshold': self.best_threshold,
            'metrics': self.metrics
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
        logger.info(f"Modelo salvo em: {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        """
        Carrega modelo salvo

        Args:
            filepath: Caminho do modelo

        Returns:
            Instância de ChurnModel
        """
        model_data = joblib.load(filepath)

        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.best_threshold = model_data['best_threshold']
        instance.metrics = model_data['metrics']
        instance.is_fitted = True

        logger.info(f"Modelo carregado de: {filepath}")
        return instance


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).parent.parent))

    from data.data_loader import DataLoader
    from data.preprocessor import DataPreprocessor
    from data.feature_engineering import FeatureEngineer

    loader = DataLoader()
    df = loader.load_data()
    X, y = loader.split_features_target()

    engineer = FeatureEngineer()
    X_engineered = engineer.create_all_features(X)

    preprocessor = DataPreprocessor()
    X_processed = preprocessor.preprocess_pipeline(X_engineered)

    model = ChurnModel(model_type='xgboost')
    metrics = model.train(X_processed, y)

    print("\n=== Métricas de Treino ===")
    print(f"Train ROC-AUC: {metrics['train']['roc_auc']:.4f}")
    print(f"Val ROC-AUC: {metrics['validation']['roc_auc']:.4f}")

    print("\n=== Feature Importance ===")
    print(model.get_feature_importance(10))
