"""
Script de teste rápido para validar o pipeline completo
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer

def test_data_loading():
    """Testa carregamento de dados"""
    print("=" * 60)
    print("TESTE 1: Carregamento de Dados")
    print("=" * 60)

    loader = DataLoader()
    df = loader.load_data()

    print(f"[OK] Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")

    validation = loader.validate_data()
    print(f"[OK] Taxa de churn: {validation['churn_rate']:.2%}")

    X, y = loader.split_features_target()
    print(f"[OK] Features: {X.shape[1]}, Target: {y.shape[0]}")
    print()

    return X, y

def test_feature_engineering(X):
    """Testa feature engineering"""
    print("=" * 60)
    print("TESTE 2: Feature Engineering")
    print("=" * 60)

    engineer = FeatureEngineer()
    X_engineered = engineer.create_all_features(X)

    print(f"[OK] Features originais: {X.shape[1]}")
    print(f"[OK] Features apos engineering: {X_engineered.shape[1]}")
    print(f"[OK] Novas features criadas: {len(engineer.feature_names)}")
    print(f"\nExemplos de features criadas:")
    for feat in engineer.feature_names[:10]:
        print(f"  - {feat}")
    print()

    return X_engineered

def test_preprocessing(X):
    """Testa preprocessamento"""
    print("=" * 60)
    print("TESTE 3: Preprocessamento")
    print("=" * 60)

    preprocessor = DataPreprocessor()
    X_processed = preprocessor.preprocess_pipeline(X, fit=True)

    print(f"[OK] Shape antes: {X.shape}")
    print(f"[OK] Shape depois: {X_processed.shape}")
    print(f"[OK] Missing values: {X_processed.isnull().sum().sum()}")
    print(f"[OK] Tipos de dados: {X_processed.dtypes.value_counts().to_dict()}")
    print()

    return X_processed

def test_model_import():
    """Testa importação do modelo"""
    print("=" * 60)
    print("TESTE 4: Importação do Modelo")
    print("=" * 60)

    try:
        from src.models.churn_model import ChurnModel
        print("[OK] ChurnModel importado com sucesso")

        model = ChurnModel(model_type='xgboost')
        print(f"[OK] Modelo XGBoost inicializado")
        print(f"[OK] Tipo: {model.model_type}")
        print()
        return True
    except Exception as e:
        print(f"[ERRO] Erro ao importar modelo: {str(e)}")
        return False

def main():
    """Executa todos os testes"""
    print("\n" + "=" * 60)
    print("TESTE COMPLETO DO PIPELINE DE CHURN")
    print("=" * 60 + "\n")

    try:
        X, y = test_data_loading()

        X_engineered = test_feature_engineering(X)

        X_processed = test_preprocessing(X_engineered)

        test_model_import()

        print("=" * 60)
        print("[SUCCESS] TODOS OS TESTES PASSARAM COM SUCESSO!")
        print("=" * 60)
        print("\nProximos passos:")
        print("1. Execute 'streamlit run app.py' para ver o dashboard")
        print("2. Treine o modelo com os notebooks em /notebooks")
        print("3. Explore as analises no dashboard interativo")
        print()

    except Exception as e:
        print("\n" + "=" * 60)
        print("[ERRO] ERRO NOS TESTES")
        print("=" * 60)
        print(f"Erro: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
