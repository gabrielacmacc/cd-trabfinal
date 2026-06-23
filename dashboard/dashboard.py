# dashboard.py

import altair as alt
import pandas as pd
import streamlit as st
from datetime import datetime
import os

alt.data_transformers.disable_max_rows()

st.set_page_config(
    page_title="Dengue Prediction Dashboard - RS/BRA",
    page_icon="🦟",
    layout="wide"
)

st.title("🦟 Dengue Prediction Dashboard")
st.caption(f"Rio Grande do Sul, Brasil | {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# ============================================================================
# CARREGAR DADOS - CAMINHO ABSOLUTO
# ============================================================================

@st.cache_data
def load_data():
    """Carrega os dados processados."""
    
    # caminho dos parquets modificar conforme a maquina ex /home/jcp/Documentos/cd-trabfinal/notebooks/data/processed/
    data_dir = '.../notebooks/data/processed/'
    
    try:
        inmet = pd.read_parquet(os.path.join(data_dir, 'inmet_weekly.parquet'))
        dengue = pd.read_parquet(os.path.join(data_dir, 'dengue_weekly.parquet'))
        snis = pd.read_parquet(os.path.join(data_dir, 'snis_combined.parquet'))
        sinisa = pd.read_parquet(os.path.join(data_dir, 'sinisa_prep.parquet'))
        
        st.success("Dados carregados com sucesso!")
        return inmet, dengue, snis, sinisa
        
    except FileNotFoundError as e:
        st.error(f"Arquivo nao encontrado: {e}")
        st.info(
            "Caminho procurado: " + data_dir + "\n\n"
            "Verifique se os arquivos existem executando no terminal:\n"
            "ls -la " + data_dir
        )
        return None, None, None, None

inmet, dengue, snis, sinisa = load_data()

if inmet is None:
    st.stop()

# ============================================================================
# MÉTRICAS RÁPIDAS
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Periodo INMET", f"{inmet['ano'].min()} - {inmet['ano'].max()}")

with col2:
    st.metric("Estacoes", inmet['estacao'].nunique())

with col3:
    st.metric("Casos de Dengue", f"{dengue['casos'].sum():,.0f}")

with col4:
    n_outbreaks = dengue['outbreak'].sum()
    st.metric("Semanas com Surto", f"{n_outbreaks} ({n_outbreaks/len(dengue)*100:.1f}%)")

with col5:
    st.metric("Municipios", dengue['localidade_id'].nunique())

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================

tab2, tab3 = st.tabs(["Dengue", "Qualidade dos Dados"])

# ============================================================================
# TAB 2: DENGUE
# ============================================================================

with tab2:
    st.header("Dados de Dengue - InfoDengue")
    st.caption(f"{len(dengue):,} registros de {dengue['localidade_id'].nunique()} municipios")

    anos_dengue = st.multiselect(
        "Selecione os Anos",
        options=sorted(dengue['ano'].unique()),
        default=sorted(dengue['ano'].unique()),
        key="anos_dengue"
    )

    filtered_dengue = dengue[dengue['ano'].isin(anos_dengue)]

    if filtered_dengue.empty:
        st.warning("Nenhum dado encontrado com os filtros selecionados.")
    else:
        filtered_dengue['data'] = pd.to_datetime(
            filtered_dengue['ano'].astype(str) + '-01-01'
        ) + pd.to_timedelta((filtered_dengue['semana_epi'] - 1) * 7, unit='D')

        st.subheader("Casos de Dengue por Semana")
        chart = alt.Chart(filtered_dengue).mark_line(color='darkred').encode(
            x=alt.X('data:T', title='Data'),
            y=alt.Y('casos:Q', title='Casos'),
            tooltip=['data', 'casos']
        ).properties(height=300).interactive()
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Casos Totais por Ano")
        casos_ano = filtered_dengue.groupby('ano')['casos'].sum().reset_index()
        chart = alt.Chart(casos_ano).mark_bar(color='coral').encode(
            x=alt.X('ano:O', title='Ano'),
            y=alt.Y('casos:Q', title='Total de Casos'),
            tooltip=['ano', 'casos']
        ).properties(height=250)
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Surtos (20 casos/100k)")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total de Surtos", filtered_dengue['outbreak'].sum())
        with col2:
            st.metric("Proporcao", f"{filtered_dengue['outbreak'].mean()*100:.1f}%")

        st.subheader("Sazonalidade dos Surtos")
        seasonality = filtered_dengue.groupby('semana_epi')['outbreak'].mean().reset_index()
        seasonality['pct'] = seasonality['outbreak'] * 100

        chart = alt.Chart(seasonality).mark_bar(color='darkblue').encode(
            x=alt.X('semana_epi:O', title='Semana Epidemiologica'),
            y=alt.Y('pct:Q', title='Proporcao de Surtos (%)'),
            tooltip=['semana_epi', 'pct']
        ).properties(height=250).interactive()
        st.altair_chart(chart, use_container_width=True)

# ============================================================================
# TAB 3: QUALIDADE
# ============================================================================

with tab3:
    st.header("Qualidade dos Dados - Missing Values")

    datasets = {
        'INMET': inmet,
        'Dengue': dengue,
        'SNIS': snis,
        'SINISA': sinisa
    }

    for name, df in datasets.items():
        if df is not None and not df.empty:
            n_missing = df.isnull().sum().sum()
            total_cells = len(df) * len(df.columns)
            pct = n_missing / total_cells * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{name} - Registros", f"{len(df):,}")
            with col2:
                st.metric(f"{name} - Missing", f"{n_missing:,}")
            with col3:
                status = "✅" if pct == 0 else "⚠️" if pct < 5 else "❌"
                st.metric(f"{name} - Status", f"{pct:.2f}% {status}")

            missing_cols = df.columns[df.isnull().any()].tolist()
            if missing_cols:
                with st.expander(f"Colunas com Missing em {name}"):
                    missing_data = pd.DataFrame({
                        'Coluna': missing_cols,
                        'Missing': [df[col].isnull().sum() for col in missing_cols],
                        '%': [(df[col].isnull().sum() / len(df)) * 100 for col in missing_cols]
                    }).sort_values('Missing', ascending=False)
                    st.dataframe(missing_data, use_container_width=True)

st.markdown("---")
st.caption("Projeto: Predicting Dengue Outbreaks from Urban Environmental Conditions in RS/BRA")