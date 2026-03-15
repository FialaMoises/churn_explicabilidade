# 🎯 Churn Prediction with Explainability & Cost Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Sistema completo de predição de churn com explicabilidade, análise de custos e simulação de ROI**

[Features](#-features) • [Instalação](#-instalação) • [Uso](#-uso) • [Dashboard](#-dashboard) • [Arquitetura](#-arquitetura)

</div>

---

## 📋 Sobre o Projeto

Este projeto implementa um **sistema end-to-end de predição de churn** que vai além da simples classificação binária. Ele fornece:

- 🤖 **Modelos de ML de alta performance** (XGBoost, LightGBM, Random Forest)
- 💡 **Explicabilidade completa** com SHAP e LIME
- 💰 **Análise de custos** com otimização de threshold baseada em custos reais
- 🎯 **Simulação de estratégias de retenção** com cálculo de ROI
- 📊 **Dashboard interativo** com Streamlit

### Problema de Negócio

Perder clientes é caro. Adquirir novos clientes custa 5-25x mais do que reter existentes. Este sistema ajuda a:

1. **Identificar** clientes em risco de churn ANTES que eles saiam
2. **Explicar** POR QUE cada cliente está em risco
3. **Priorizar** ações baseadas no valor do cliente (CLV)
4. **Otimizar** orçamento de retenção com simulações de ROI
5. **Medir** impacto financeiro de cada decisão

---

## ✨ Features

### 1. Feature Engineering Avançado
- **40+ features engineered** incluindo:
  - Features de tenure e loyalty
  - Financial scores compostos
  - Engagement metrics
  - Risk scores
  - Customer value segments

### 2. Modelos de Machine Learning
- Logistic Regression (baseline interpretável)
- Random Forest
- **XGBoost** (melhor performance)
- LightGBM
- Ensemble methods

### 3. Explicabilidade (XAI)
- **SHAP Values**: Explicações globais e locais
- **LIME**: Explicações interpretáveis por cliente
- Feature importance multi-perspectiva
- Partial Dependence Plots

### 4. Análise de Custo-Benefício
- Matriz de custos customizável
- Otimização de threshold baseada em custo
- Segmentação de valor do cliente
- Expected value framework

### 5. Simulação de Retenção
- Monte Carlo simulation (10k+ iterações)
- Múltiplas estratégias (Top Risk, High Value, Segmented, Preventive)
- Análise de cenários (Otimista, Realista, Pessimista)
- ROI projetado com intervalos de confiança

### 6. Dashboard Interativo
- **Streamlit app** com visualizações interativas
- 6 páginas principais:
  - 🏠 Overview
  - 📈 Análise Exploratória
  - 🤖 Modelo & Predições
  - 💡 Explicabilidade
  - 💰 Análise de Custo
  - 🎯 Simulação de Retenção

---

## 🚀 Instalação

### Opção 1: Docker (Recomendado) 🐳

#### Pré-requisitos
- Docker Desktop instalado
- Docker Compose

#### Início Rápido

**Windows:**
```batch
# Iniciar todos os serviços
run.bat start

# Acessar:
# - Dashboard: http://localhost:8501
# - Jupyter Lab: http://localhost:8888
```

**Linux/Mac:**
```bash
# Dar permissão de execução
chmod +x run.sh

# Iniciar todos os serviços
./run.sh start

# Acessar:
# - Dashboard: http://localhost:8501
# - Jupyter Lab: http://localhost:8888
```

#### Comandos Docker Disponíveis

```bash
./run.sh start    # Iniciar containers
./run.sh stop     # Parar containers
./run.sh restart  # Reiniciar containers
./run.sh logs     # Ver logs em tempo real
./run.sh build    # Reconstruir imagens
./run.sh clean    # Limpar tudo
./run.sh shell    # Abrir shell no container
./run.sh test     # Executar testes
```

---

### Opção 2: Instalação Local

#### Pré-requisitos
- Python 3.8+
- pip

#### Passo a Passo

```bash
# 1. Clone o repositório
git clone https://github.com/seu-usuario/churn_explicabilidade.git
cd churn_explicabilidade

# 2. Crie um ambiente virtual (recomendado)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Baixe o dataset (se não baixado ainda)
python ../download_dataset.py

# 5. Execute o dashboard
streamlit run app.py
```

---

## 💻 Uso

### 🐳 Com Docker (Recomendado)

```bash
# Iniciar dashboard
./run.sh start

# Acessar http://localhost:8501

# Executar testes
./run.sh test
```

### 💻 Sem Docker

#### 1. Executar o Dashboard

```bash
streamlit run app.py
```

O dashboard abrirá automaticamente em `http://localhost:8501`

#### 2. Executar Testes

```bash
python test_pipeline.py
```

#### 3. Executar Scripts Python

##### Carregar e Explorar Dados
```python
from src.data.data_loader import DataLoader

loader = DataLoader()
df = loader.load_data()
X, y = loader.split_features_target()

print(loader.get_feature_info())
```

#### Feature Engineering
```python
from src.data.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
X_engineered = engineer.create_all_features(X)

print(f"Features criadas: {len(engineer.feature_names)}")
```

#### Treinar Modelo
```python
from src.models.churn_model import ChurnModel

model = ChurnModel(model_type='xgboost')
metrics = model.train(X_engineered, y)

print(f"ROC-AUC: {metrics['validation']['roc_auc']:.4f}")
```

---

## 📊 Dashboard

### Screenshots

#### 🏠 Overview
- KPIs principais (Total clientes, Taxa de churn, Ativos vs Churned)
- Distribuição de churn por geografia
- Estatísticas do dataset

#### 📈 Análise Exploratória
- Distribuições de variáveis
- Matriz de correlação
- Análise por segmentos

#### 🤖 Modelo & Predições
- Arquitetura do modelo
- Métricas de performance

#### 💡 Explicabilidade
- SHAP values
- LIME explanations
- Feature importance

#### 💰 Análise de Custo
- Matriz de custos
- Otimização de threshold

#### 🎯 Simulação de Retenção
- Simulador interativo de ROI
- Comparação de estratégias

---

## 📊 Dataset

**Fonte**: [Bank Customer Churn - Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)

### Estatísticas

- **Total de clientes**: ~10.000
- **Taxa de churn**: ~20%
- **Features**: 18 originais → 40+ após feature engineering

---

## 🎯 Resultados Esperados

### Métricas de Modelo
- **ROC-AUC**: > 0.85
- **Precision @ Top 10%**: > 0.70
- **F1-Score**: > 0.75

### Impacto de Negócio
- **Redução de churn**: 20-40%
- **ROI de campanhas**: 2.5x - 4.0x

---

## 🛠 Tecnologias Utilizadas

- Python 3.8+
- Pandas, NumPy, Scikit-learn
- XGBoost, LightGBM
- SHAP, LIME
- Streamlit, Plotly

---

<div align="center">

**Desenvolvido com ❤️ para demonstrar skills em ML, XAI e Data Science**

⭐ Se este projeto foi útil, considere dar uma estrela!

</div>