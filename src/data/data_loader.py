"""
Data Loader Module
Responsável por carregar e validar dados do projeto de churn
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Classe para carregar e validar dados de churn"""

    def __init__(self, data_path: Optional[str] = None):
        """
        Inicializa o DataLoader

        Args:
            data_path: Caminho para o arquivo de dados
        """
        if data_path is None:
            self.data_path = Path(__file__).parent.parent.parent / "dataraw" / "Customer-Churn-Records.csv"
        else:
            self.data_path = Path(data_path)

        self.df = None

    def load_data(self) -> pd.DataFrame:
        """
        Carrega os dados do arquivo CSV

        Returns:
            DataFrame com os dados carregados
        """
        try:
            logger.info(f"Carregando dados de: {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Dados carregados com sucesso: {self.df.shape[0]} linhas, {self.df.shape[1]} colunas")
            return self.df
        except FileNotFoundError:
            logger.error(f"Arquivo não encontrado: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            raise

    def validate_data(self) -> dict:
        """
        Valida a qualidade dos dados

        Returns:
            Dicionário com estatísticas de validação
        """
        if self.df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")

        validation_stats = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'data_types': self.df.dtypes.to_dict(),
            'churn_rate': self.df['Exited'].mean() if 'Exited' in self.df.columns else None
        }

        logger.info("Validação de dados concluída:")
        logger.info(f"  - Total de registros: {validation_stats['total_rows']}")
        logger.info(f"  - Total de features: {validation_stats['total_columns']}")
        logger.info(f"  - Duplicatas: {validation_stats['duplicates']}")
        logger.info(f"  - Taxa de churn: {validation_stats['churn_rate']:.2%}")

        return validation_stats

    def get_feature_info(self) -> pd.DataFrame:
        """
        Retorna informações sobre as features

        Returns:
            DataFrame com informações das features
        """
        if self.df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")

        info_dict = {
            'Feature': self.df.columns,
            'Type': self.df.dtypes.values,
            'Missing': self.df.isnull().sum().values,
            'Missing %': (self.df.isnull().sum() / len(self.df) * 100).values,
            'Unique': [self.df[col].nunique() for col in self.df.columns],
            'Sample': [self.df[col].iloc[0] if len(self.df) > 0 else None for col in self.df.columns]
        }

        return pd.DataFrame(info_dict)

    def split_features_target(self, target_col: str = 'Exited',
                             drop_cols: list = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa features e target

        Args:
            target_col: Nome da coluna target
            drop_cols: Lista de colunas para remover (além do target)

        Returns:
            Tuple com (features, target)
        """
        if self.df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")

        if drop_cols is None:
            drop_cols = ['RowNumber', 'CustomerId', 'Surname']

        y = self.df[target_col]
        cols_to_drop = drop_cols + [target_col]
        X = self.df.drop(columns=cols_to_drop, errors='ignore')

        logger.info(f"Features: {X.shape[1]} colunas")
        logger.info(f"Target: {target_col}")

        return X, y

    def get_summary_stats(self) -> dict:
        """
        Retorna estatísticas resumidas dos dados

        Returns:
            Dicionário com estatísticas
        """
        if self.df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")

        stats = {
            'numerical_summary': self.df.describe(),
            'categorical_summary': self.df.describe(include=['object']),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,
        }

        return stats


if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data()

    print("\n=== Informações dos Dados ===")
    print(loader.get_feature_info())

    print("\n=== Validação ===")
    validation = loader.validate_data()

    print("\n=== Separação Features/Target ===")
    X, y = loader.split_features_target()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"\nDistribuição do Target:\n{y.value_counts(normalize=True)}")
