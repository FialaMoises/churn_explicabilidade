"""
Feature Engineering Module
Cria features avançadas para o modelo de churn
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Classe para engenharia de features"""

    def __init__(self):
        self.feature_names = []

    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features baseadas em Tenure (tempo como cliente)

        Args:
            df: DataFrame original

        Returns:
            DataFrame com novas features
        """
        df_new = df.copy()

        if 'Tenure' in df_new.columns:
            df_new['Tenure_Category'] = pd.cut(df_new['Tenure'],
                                               bins=[-1, 2, 5, 10],
                                               labels=['New', 'Regular', 'Loyal'])

            df_new['Tenure_Years'] = df_new['Tenure'] / 12

            df_new['Is_New_Customer'] = (df_new['Tenure'] <= 2).astype(int)

            df_new['Is_Loyal_Customer'] = (df_new['Tenure'] >= 7).astype(int)

            logger.info("Features de Tenure criadas")

        return df_new

    def create_balance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features baseadas em Balance

        Args:
            df: DataFrame original

        Returns:
            DataFrame com novas features
        """
        df_new = df.copy()

        if 'Balance' in df_new.columns:
            df_new['Balance_Log'] = np.log1p(df_new['Balance'])

            df_new['Has_Balance'] = (df_new['Balance'] > 0).astype(int)

            df_new['Balance_Category'] = pd.cut(df_new['Balance'],
                                                bins=[-1, 1, 50000, 100000, 150000, float('inf')],
                                                labels=['Zero', 'Low', 'Medium', 'High', 'Very_High'])

            if 'NumOfProducts' in df_new.columns:
                df_new['Balance_Per_Product'] = df_new['Balance'] / (df_new['NumOfProducts'] + 1)

            logger.info("Features de Balance criadas")

        return df_new

    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features financeiras combinadas

        Args:
            df: DataFrame original

        Returns:
            DataFrame com novas features
        """
        df_new = df.copy()

        if 'Balance' in df_new.columns and 'EstimatedSalary' in df_new.columns:
            df_new['Balance_Salary_Ratio'] = df_new['Balance'] / (df_new['EstimatedSalary'] + 1)

        if 'Balance' in df_new.columns and 'EstimatedSalary' in df_new.columns:
            df_new['Is_High_Net_Worth'] = (
                (df_new['Balance'] > 100000) & (df_new['EstimatedSalary'] > 100000)
            ).astype(int)

        if all(col in df_new.columns for col in ['CreditScore', 'Balance', 'EstimatedSalary']):
            credit_norm = (df_new['CreditScore'] - df_new['CreditScore'].min()) / \
                         (df_new['CreditScore'].max() - df_new['CreditScore'].min())
            balance_norm = (df_new['Balance'] - df_new['Balance'].min()) / \
                          (df_new['Balance'].max() - df_new['Balance'].min())
            salary_norm = (df_new['EstimatedSalary'] - df_new['EstimatedSalary'].min()) / \
                         (df_new['EstimatedSalary'].max() - df_new['EstimatedSalary'].min())

            df_new['Financial_Score'] = (credit_norm + balance_norm + salary_norm) / 3

        logger.info("Features financeiras criadas")

        return df_new

    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features demográficas

        Args:
            df: DataFrame original

        Returns:
            DataFrame com novas features
        """
        df_new = df.copy()

        if 'Age' in df_new.columns:
            df_new['Age_Group'] = pd.cut(df_new['Age'],
                                        bins=[0, 30, 40, 50, 60, 100],
                                        labels=['Young', 'Adult', 'Middle_Age', 'Senior', 'Elderly'])

            df_new['Is_Risk_Age'] = ((df_new['Age'] < 25) | (df_new['Age'] > 60)).astype(int)

        if 'Geography' in df_new.columns and 'Gender' in df_new.columns:
            df_new['Geography_Gender'] = df_new['Geography'].astype(str) + '_' + df_new['Gender'].astype(str)

        logger.info("Features demográficas criadas")

        return df_new

    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de engajamento do cliente

        Args:
            df: DataFrame original

        Returns:
            DataFrame com novas features
        """
        df_new = df.copy()

        engagement_components = []

        if 'IsActiveMember' in df_new.columns:
            engagement_components.append(df_new['IsActiveMember'])

        if 'HasCrCard' in df_new.columns:
            engagement_components.append(df_new['HasCrCard'])

        if 'NumOfProducts' in df_new.columns:
            products_norm = (df_new['NumOfProducts'] - 1) / 3
            engagement_components.append(products_norm)

        if engagement_components:
            df_new['Engagement_Score'] = sum(engagement_components) / len(engagement_components)

        if 'IsActiveMember' in df_new.columns and 'NumOfProducts' in df_new.columns:
            df_new['Is_Low_Engagement'] = (
                (df_new['IsActiveMember'] == 0) & (df_new['NumOfProducts'] == 1)
            ).astype(int)

        if 'IsActiveMember' in df_new.columns and 'NumOfProducts' in df_new.columns and 'HasCrCard' in df_new.columns:
            df_new['Is_High_Engagement'] = (
                (df_new['IsActiveMember'] == 1) &
                (df_new['NumOfProducts'] >= 2) &
                (df_new['HasCrCard'] == 1)
            ).astype(int)

        logger.info("Features de engajamento criadas")

        return df_new

    def create_satisfaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features relacionadas à satisfação

        Args:
            df: DataFrame original

        Returns:
            DataFrame com novas features
        """
        df_new = df.copy()

        if 'Satisfaction Score' in df_new.columns:
            df_new['Is_Unsatisfied'] = (df_new['Satisfaction Score'] <= 2).astype(int)
            df_new['Is_Very_Satisfied'] = (df_new['Satisfaction Score'] >= 4).astype(int)

        if 'Complain' in df_new.columns and 'Satisfaction Score' in df_new.columns:
            df_new['Complain_Low_Satisfaction'] = (
                (df_new['Complain'] == 1) & (df_new['Satisfaction Score'] <= 2)
            ).astype(int)

        if 'Complain' in df_new.columns and 'Satisfaction Score' in df_new.columns:
            satisfaction_risk = (5 - df_new['Satisfaction Score']) / 4
            complaint_risk = df_new['Complain']
            df_new['Satisfaction_Risk_Score'] = (satisfaction_risk + complaint_risk) / 2

        logger.info("Features de satisfação criadas")

        return df_new

    def create_customer_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de valor do cliente (CLV proxy)

        Args:
            df: DataFrame original

        Returns:
            DataFrame com novas features
        """
        df_new = df.copy()

        if all(col in df_new.columns for col in ['Balance', 'EstimatedSalary', 'Tenure', 'NumOfProducts']):
            df_new['Estimated_CLV'] = (
                df_new['Balance'] * 0.3 +
                df_new['EstimatedSalary'] * 0.2 +
                df_new['Tenure'] * 1000 +
                df_new['NumOfProducts'] * 5000
            )

            df_new['Customer_Value_Segment'] = pd.qcut(
                df_new['Estimated_CLV'],
                q=3,
                labels=['Low_Value', 'Medium_Value', 'High_Value'],
                duplicates='drop'
            )

        if 'NumOfProducts' in df_new.columns:
            df_new['Estimated_Annual_Revenue'] = df_new['NumOfProducts'] * 2000

        logger.info("Features de valor do cliente criadas")

        return df_new

    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de risco de churn

        Args:
            df: DataFrame original

        Returns:
            DataFrame com novas features
        """
        df_new = df.copy()

        risk_flags = []

        if 'IsActiveMember' in df_new.columns:
            df_new['Risk_Inactive'] = (df_new['IsActiveMember'] == 0).astype(int)
            risk_flags.append('Risk_Inactive')

        if 'Balance' in df_new.columns:
            df_new['Risk_Zero_Balance'] = (df_new['Balance'] == 0).astype(int)
            risk_flags.append('Risk_Zero_Balance')

        if 'NumOfProducts' in df_new.columns:
            df_new['Risk_Single_Product'] = (df_new['NumOfProducts'] == 1).astype(int)
            risk_flags.append('Risk_Single_Product')

        if 'CreditScore' in df_new.columns:
            df_new['Risk_Low_Credit'] = (df_new['CreditScore'] < 600).astype(int)
            risk_flags.append('Risk_Low_Credit')

        if 'Complain' in df_new.columns:
            df_new['Risk_Complaint'] = df_new['Complain']
            risk_flags.append('Risk_Complaint')

        if risk_flags:
            df_new['Composite_Risk_Score'] = df_new[risk_flags].sum(axis=1) / len(risk_flags)

        logger.info("Features de risco criadas")

        return df_new

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria todas as features de uma vez

        Args:
            df: DataFrame original

        Returns:
            DataFrame com todas as features
        """
        logger.info("Iniciando criação de todas as features...")

        df_engineered = df.copy()

        df_engineered = self.create_tenure_features(df_engineered)
        df_engineered = self.create_balance_features(df_engineered)
        df_engineered = self.create_financial_features(df_engineered)
        df_engineered = self.create_demographic_features(df_engineered)
        df_engineered = self.create_engagement_features(df_engineered)
        df_engineered = self.create_satisfaction_features(df_engineered)
        df_engineered = self.create_customer_value_features(df_engineered)
        df_engineered = self.create_risk_features(df_engineered)

        self.feature_names = [col for col in df_engineered.columns if col not in df.columns]

        logger.info(f"Feature engineering concluído! {len(self.feature_names)} novas features criadas")
        logger.info(f"Features criadas: {self.feature_names[:10]}...")

        return df_engineered


if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader()
    df = loader.load_data()
    X, y = loader.split_features_target()

    engineer = FeatureEngineer()
    X_engineered = engineer.create_all_features(X)

    print("\n=== Feature Engineering Concluído ===")
    print(f"Shape original: {X.shape}")
    print(f"Shape com novas features: {X_engineered.shape}")
    print(f"\nNovas features criadas ({len(engineer.feature_names)}):")
    for feat in engineer.feature_names:
        print(f"  - {feat}")
