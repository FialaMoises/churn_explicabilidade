"""
Dashboard Streamlit - Churn Prediction & Explainability
Front-end interativo para análise de churn
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer

st.set_page_config(
    page_title="Churn Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Carrega os dados com cache"""
    loader = DataLoader()
    df = loader.load_data()
    X, y = loader.split_features_target()
    return df, X, y


@st.cache_data
def engineer_features(X):
    """Aplica feature engineering com cache"""
    engineer = FeatureEngineer()
    X_engineered = engineer.create_all_features(X)
    return X_engineered


def main():
    st.markdown('<div class="main-header">📊 Churn Prediction & Explainability Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/business-report.png", width=100)
        st.title("⚙️ Configurações")

        page = st.selectbox(
            "Navegação",
            ["🏠 Overview", "📈 Análise Exploratória", "🤖 Modelo & Predições",
             "💡 Explicabilidade", "💰 Análise de Custo", "🎯 Simulação de Retenção"]
        )

        st.markdown("---")
        st.markdown("### Sobre")
        st.info("""
        **Churn Prediction System**

        Sistema completo de predição de churn com:
        - Análise exploratória
        - Modelos de ML
        - Explicabilidade (SHAP/LIME)
        - Análise de custos
        - Simulação de ROI
        """)

    try:
        df, X, y = load_data()
        df_full = df.copy()
        df_full['Exited'] = y
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        st.stop()

    if page == "🏠 Overview":
        show_overview(df_full, X, y)
    elif page == "📈 Análise Exploratória":
        show_eda(df_full, X, y)
    elif page == "🤖 Modelo & Predições":
        show_model_predictions(df_full, X, y)
    elif page == "💡 Explicabilidade":
        show_explainability(df_full, X, y)
    elif page == "💰 Análise de Custo":
        show_cost_analysis(df_full, X, y)
    elif page == "🎯 Simulação de Retenção":
        show_retention_simulation(df_full, X, y)


def show_overview(df, X, y):
    """Página de Overview"""
    st.header("🏠 Overview do Dataset")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total de Clientes", f"{len(df):,}")
    with col2:
        churn_rate = y.mean() * 100
        st.metric("Taxa de Churn", f"{churn_rate:.1f}%",
                 delta=f"{churn_rate-20:.1f}% vs target",
                 delta_color="inverse")
    with col3:
        st.metric("Clientes Ativos", f"{(y==0).sum():,}")
    with col4:
        st.metric("Clientes Churned", f"{(y==1).sum():,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Distribuição de Churn")
        fig = px.pie(
            values=[int((y==0).sum()), int((y==1).sum())],
            names=['Ativo', 'Churned'],
            color_discrete_sequence=['#2ecc71', '#e74c3c'],
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🌍 Churn por Geografia")
        if 'Geography' in df.columns:
            churn_geo = df.groupby('Geography')['Exited'].agg(['sum', 'count', 'mean']).reset_index()
            churn_geo['rate'] = churn_geo['mean'] * 100

            fig = px.bar(
                churn_geo,
                x='Geography',
                y='rate',
                color='rate',
                color_continuous_scale='Reds',
                labels={'rate': 'Taxa de Churn (%)'}
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Estatísticas do Dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Informações Gerais**")
        info_df = pd.DataFrame({
            'Métrica': ['Total de Registros', 'Total de Features', 'Features Numéricas', 'Features Categóricas'],
            'Valor': [
                len(df),
                len(df.columns),
                len(df.select_dtypes(include=[np.number]).columns),
                len(df.select_dtypes(include=['object']).columns)
            ]
        })
        st.dataframe(info_df, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**Missing Values**")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            st.dataframe(missing.to_frame('Missing Count'), use_container_width=True)
        else:
            st.success("✅ Nenhum valor missing encontrado!")

    st.subheader("🔍 Prévia dos Dados")
    st.dataframe(df.head(20), use_container_width=True)


def show_eda(df, X, y):
    """Página de Análise Exploratória"""
    st.header("📈 Análise Exploratória de Dados")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Distribuições", "📈 Correlações", "👥 Segmentação", "⏰ Análise Temporal"
    ])

    with tab1:
        st.subheader("Distribuições das Variáveis")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                df, x='Age', color='Exited',
                marginal='box',
                nbins=30,
                color_discrete_sequence=['#2ecc71', '#e74c3c'],
                labels={'Exited': 'Churn'}
            )
            fig.update_layout(title="Distribuição de Idade", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                df[df['Balance'] > 0], x='Balance', color='Exited',
                marginal='box',
                nbins=30,
                color_discrete_sequence=['#2ecc71', '#e74c3c'],
                labels={'Exited': 'Churn'}
            )
            fig.update_layout(title="Distribuição de Balance (> 0)", height=400)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig = px.histogram(
                df, x='CreditScore', color='Exited',
                marginal='box',
                nbins=30,
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            fig.update_layout(title="Distribuição de Credit Score", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            fig = px.histogram(
                df, x='EstimatedSalary', color='Exited',
                marginal='box',
                nbins=30,
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            fig.update_layout(title="Distribuição de Salário Estimado", height=400)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Matriz de Correlação")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = df[numeric_cols].corr()

        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlação"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Análise por Segmentos")

        col1, col2 = st.columns(2)

        with col1:
            if 'Gender' in df.columns:
                gender_churn = df.groupby(['Gender', 'Exited']).size().reset_index(name='Count')
                fig = px.bar(
                    gender_churn, x='Gender', y='Count', color='Exited',
                    barmode='group',
                    color_discrete_sequence=['#2ecc71', '#e74c3c'],
                    labels={'Exited': 'Churn'}
                )
                fig.update_layout(title="Churn por Gênero", height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'NumOfProducts' in df.columns:
                products_churn = df.groupby(['NumOfProducts', 'Exited']).size().reset_index(name='Count')
                fig = px.bar(
                    products_churn, x='NumOfProducts', y='Count', color='Exited',
                    barmode='group',
                    color_discrete_sequence=['#2ecc71', '#e74c3c'],
                    labels={'Exited': 'Churn'}
                )
                fig.update_layout(title="Churn por Número de Produtos", height=400)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Taxa de Churn por Segmento")

        segments = []
        if 'Gender' in df.columns:
            gender_rate = df.groupby('Gender')['Exited'].mean() * 100
            for gender, rate in gender_rate.items():
                segments.append({'Segmento': f'Gender: {gender}', 'Taxa de Churn (%)': f'{rate:.2f}'})

        if 'Geography' in df.columns:
            geo_rate = df.groupby('Geography')['Exited'].mean() * 100
            for geo, rate in geo_rate.items():
                segments.append({'Segmento': f'Geography: {geo}', 'Taxa de Churn (%)': f'{rate:.2f}'})

        if segments:
            st.dataframe(pd.DataFrame(segments), hide_index=True, use_container_width=True)

    with tab4:
        st.subheader("Análise de Tenure (Tempo como Cliente)")

        if 'Tenure' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(
                    df, x='Tenure', color='Exited',
                    nbins=11,
                    color_discrete_sequence=['#2ecc71', '#e74c3c'],
                    labels={'Exited': 'Churn'}
                )
                fig.update_layout(title="Distribuição de Tenure", height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                tenure_churn = df.groupby('Tenure')['Exited'].mean().reset_index()
                tenure_churn['Churn Rate (%)'] = tenure_churn['Exited'] * 100

                fig = px.line(
                    tenure_churn, x='Tenure', y='Churn Rate (%)',
                    markers=True,
                    color_discrete_sequence=['#e74c3c']
                )
                fig.update_layout(title="Taxa de Churn por Tenure", height=400)
                st.plotly_chart(fig, use_container_width=True)


def show_model_predictions(df, X, y):
    """Página de Modelo e Predições"""
    st.header("🤖 Modelo & Predições")

    st.info("ℹ️ Esta seção permite treinar modelos de ML e fazer predições. Para treinar o modelo, execute o notebook de treinamento primeiro.")

    st.markdown("### Arquitetura do Modelo")
    st.markdown("""
    **Pipeline de ML:**
    1. Feature Engineering (40+ features)
    2. Preprocessing (encoding, scaling)
    3. Model Training (XGBoost/LightGBM/Random Forest)
    4. Calibration (probabilidades)
    5. Threshold Optimization (custo-baseado)
    """)

    st.markdown("### Métricas Esperadas")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC-AUC", "> 0.85", delta="Target")
    with col2:
        st.metric("Precision@Top10%", "> 0.70", delta="Target")
    with col3:
        st.metric("F1-Score", "> 0.75", delta="Target")
    with col4:
        st.metric("Brier Score", "< 0.15", delta="Target")


def show_explainability(df, X, y):
    """Página de Explicabilidade"""
    st.header("💡 Explicabilidade do Modelo")

    st.info("ℹ️ Esta seção mostra como o modelo toma decisões usando SHAP e LIME. Execute o notebook de explicabilidade primeiro.")

    st.markdown("### Por que Explicabilidade?")
    st.markdown("""
    - **Transparência**: Entender quais fatores levam ao churn
    - **Ação**: Saber O QUE fazer para reter clientes
    - **Confiança**: Stakeholders confiam no modelo
    - **Regulação**: Compliance com regulações (GDPR, etc.)
    """)

    st.markdown("### Técnicas Usadas")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**SHAP (SHapley Additive exPlanations)**")
        st.markdown("""
        - Feature importance global
        - Contribuição individual por cliente
        - Interações entre features
        - Baseado em teoria dos jogos
        """)

    with col2:
        st.markdown("**LIME (Local Interpretable Model-agnostic Explanations)**")
        st.markdown("""
        - Explicações locais (por cliente)
        - Model-agnostic
        - Fácil interpretação
        - Validação cruzada com SHAP
        """)


def show_cost_analysis(df, X, y):
    """Página de Análise de Custo"""
    st.header("💰 Análise de Custo de Erro")

    st.markdown("### Matriz de Custos")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Falso Positivo (FP)**")
        st.error("""
        **Custo: R$ 100**
        - Cliente não iria sair
        - Recebeu incentivo desnecessário
        - Desconto na margem
        """)

        st.markdown("**Verdadeiro Negativo (TN)**")
        st.success("""
        **Custo: R$ 0**
        - Cliente não iria sair
        - Não recebeu incentivo
        - Sem custo
        """)

    with col2:
        st.markdown("**Falso Negativo (FN)**")
        st.error("""
        **Custo: R$ 2.000**
        - Cliente saiu (perdido)
        - Não foi identificado
        - LTV perdido + custo aquisição
        """)

        st.markdown("**Verdadeiro Positivo (TP)**")
        st.success("""
        **Valor: R$ 500 (líquido)**
        - Cliente identificado
        - Campanha executada
        - Cliente retido (40% taxa sucesso)
        """)

    st.markdown("---")
    st.markdown("### Otimização de Threshold")
    st.markdown("""
    O threshold ótimo minimiza o custo total esperado:

    **Custo Total = (FP × Custo_FP) + (FN × Custo_FN) - (TP × Valor_TP)**

    Em vez de usar threshold=0.5, otimizamos baseado no custo real do negócio.
    """)


def show_retention_simulation(df, X, y):
    """Página de Simulação de Retenção"""
    st.header("🎯 Simulação de Estratégias de Retenção")

    st.markdown("### Estratégias Disponíveis")

    strategies = pd.DataFrame({
        'Estratégia': ['Top 10% Risco', 'Alto Valor em Risco', 'Segmentada', 'Preventiva'],
        'Clientes Alvo': [int(len(df)*0.1), int(len(df)*0.05), int(len(df)*0.15), int(len(df)*0.20)],
        'Custo Médio': ['R$ 100', 'R$ 200', 'R$ 120', 'R$ 80'],
        'Taxa Retenção': ['45%', '60%', '50%', '40%'],
        'ROI Esperado': ['2.5x', '4.0x', '3.2x', '2.0x']
    })

    st.dataframe(strategies, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Simulador de ROI")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Parâmetros**")
        budget = st.slider("Orçamento Total (R$)", 10000, 500000, 100000, 10000)
        target_pct = st.slider("% Clientes Alvo", 5, 40, 10, 5)
        retention_rate = st.slider("Taxa de Retenção (%)", 20, 80, 40, 5)
        avg_ltv = st.slider("LTV Médio (R$)", 500, 5000, 2000, 100)

    with col2:
        n_customers = int(len(df) * (target_pct/100))
        cost_per_customer = budget / n_customers if n_customers > 0 else 0
        customers_retained = int(n_customers * (retention_rate/100))
        total_value_saved = customers_retained * avg_ltv
        net_roi = total_value_saved - budget
        roi_percentage = (net_roi / budget * 100) if budget > 0 else 0

        st.markdown("**Resultados da Simulação**")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Clientes Alvo", f"{n_customers:,}")
            st.metric("Custo/Cliente", f"R$ {cost_per_customer:.2f}")
        with res_col2:
            st.metric("Clientes Retidos", f"{customers_retained:,}")
            st.metric("Valor Preservado", f"R$ {total_value_saved:,}")
        with res_col3:
            st.metric("ROI Líquido", f"R$ {net_roi:,}",
                     delta=f"{roi_percentage:.1f}%")
            roi_ratio = total_value_saved / budget if budget > 0 else 0
            st.metric("ROI Ratio", f"{roi_ratio:.2f}x")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Investimento', 'Retorno', 'Lucro Líquido'],
            y=[budget, total_value_saved, net_roi],
            marker_color=['#e74c3c', '#2ecc71', '#3498db']
        ))
        fig.update_layout(
            title="Análise de ROI",
            yaxis_title="Valor (R$)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
