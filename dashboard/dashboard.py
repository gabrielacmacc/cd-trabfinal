# dashboard.py

import altair as alt
import pandas as pd
import streamlit as st
from datetime import datetime
import os

alt.data_transformers.disable_max_rows()

st.set_page_config(
    page_title="Predicting Dengue Outbreaks from Urban Environmental Conditions",
    page_icon="🦟",
    layout="wide"
)

st.title("🦟 Predicting Dengue Outbreaks from Urban Environmental Conditions")
st.caption(f"Rio Grande do Sul, Brasil | {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# ============================================================================
# CARREGAR DADOS - CAMINHO ABSOLUTO
# ============================================================================

@st.cache_data
def load_data():
    """Carrega os dados processados."""
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'notebooks', 'data','processed')
    
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
    st.metric("Estacoes INMET", inmet['estacao'].nunique())

with col3:
    st.metric("Casos de Dengue", f"{dengue['casos'].sum():,.0f}")

with col4:
    n_outbreaks = dengue['outbreak'].sum()
    st.metric("Semanas com Surto", f"{n_outbreaks} ({n_outbreaks/len(dengue)*100:.1f}%)")

with col5:
    st.metric("Municipios", dengue['municipio_nome'].nunique())

st.markdown("---")

# ============================================================================
# TABS
# ============================================================================

tab2, tab3, tab5, tab4 = st.tabs(["Dengue", "Meteorologia", "Saneamento", "Qualidade dos Dados"])

# ============================================================================
# TAB 2: DENGUE
# ============================================================================

with tab2:
    st.header("Dados de Dengue - InfoDengue")
    st.caption(f"{len(dengue):,} registros de {dengue['municipio_nome'].nunique()} municipios")

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
            x=alt.X('semana_epi:O', title='Semana Epidemiológica'),
            y=alt.Y('pct:Q', title='Proporcao de Surtos (%)'),
            tooltip=['semana_epi', 'pct']
        ).properties(height=250).interactive()
        st.altair_chart(chart, use_container_width=True)

# ============================================================================
# TAB 3: INMET
# ============================================================================

with tab3:
    st.header("Dados de Meteorologia - INMET")

    anos_inmet = st.multiselect(
        "Selecione os Anos",
        options=sorted(inmet['ano'].unique()),
        default=sorted(inmet['ano'].unique()),
        key="anos_inmet"
    )

    estacoes_inmet = st.multiselect(
        "Selecione as Estações",
        options=sorted(inmet['estacao'].unique()),
        default=sorted(inmet['estacao'].unique()),
        key="estacoes_inmet"
    )

    filtered_inmet = inmet[inmet['ano'].isin(anos_inmet) & inmet['estacao'].isin(estacoes_inmet)]

    if filtered_inmet.empty:
        st.warning("Nenhum dado encontrado com os filtros selecionados.")
    else:
        filtered_inmet = filtered_inmet.copy()
        filtered_inmet['data'] = pd.to_datetime(
            filtered_inmet['ano'].astype(str) + '-01-01'
        ) + pd.to_timedelta((filtered_inmet['semana_epi'] - 1) * 7, unit='D')

        weekly = filtered_inmet.groupby('data').agg(
            temp_media=('temp_media', 'mean'),
            temp_min=('temp_min', 'mean'),
            temp_max=('temp_max', 'mean'),
            umidade_media=('umidade_media', 'mean'),
            precipitacao_total=('precipitacao_total', 'mean'),
            vento_medio=('vento_medio', 'mean'),
        ).reset_index()

        st.subheader("Temperatura Média Semanal")
        temp_band = alt.Chart(weekly).mark_area(opacity=0.2, color='tomato').encode(
            x=alt.X('data:T', title='Data'),
            y=alt.Y('temp_min:Q', title='Temperatura (°C)'),
            y2=alt.Y2('temp_max:Q')
        )
        temp_line = alt.Chart(weekly).mark_line(color='tomato').encode(
            x=alt.X('data:T', title='Data'),
            y=alt.Y('temp_media:Q', title='Temperatura (°C)'),
            tooltip=[alt.Tooltip('data:T', title='Data'), alt.Tooltip('temp_media:Q', title='Temp Media', format='.1f'),
                     alt.Tooltip('temp_min:Q', title='Temp Min', format='.1f'), alt.Tooltip('temp_max:Q', title='Temp Max', format='.1f')]
        )
        st.altair_chart((temp_band + temp_line).properties(height=300).interactive(), use_container_width=True)

        st.subheader("Precipitação Total Semanal")
        chart_prec = alt.Chart(weekly).mark_bar(color='steelblue').encode(
            x=alt.X('data:T', title='Data'),
            y=alt.Y('precipitacao_total:Q', title='Precipitacao (mm)'),
            tooltip=[alt.Tooltip('data:T', title='Data'), alt.Tooltip('precipitacao_total:Q', title='Precipitacao (mm)', format='.1f')]
        ).properties(height=250).interactive()
        st.altair_chart(chart_prec, use_container_width=True)

        st.subheader("Umidade Relativa Média Semanal")
        chart_umid = alt.Chart(weekly).mark_line(color='teal').encode(
            x=alt.X('data:T', title='Data'),
            y=alt.Y('umidade_media:Q', title='Umidade (%)', scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip('data:T', title='Data'), alt.Tooltip('umidade_media:Q', title='Umidade (%)', format='.1f')]
        ).properties(height=250).interactive()
        st.altair_chart(chart_umid, use_container_width=True)

        st.subheader("Sazonalidade por Semana Epidemiológica")
        seasonal = filtered_inmet.groupby('semana_epi').agg(
            temp_media=('temp_media', 'mean'),
            precipitacao_total=('precipitacao_total', 'mean'),
            umidade_media=('umidade_media', 'mean'),
        ).reset_index()

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.altair_chart(alt.Chart(seasonal).mark_line(color='tomato').encode(
                x=alt.X('semana_epi:O', title='Semana Epidemiológica'),
                y=alt.Y('temp_media:Q', title='Temp Media (°C)'),
                tooltip=['semana_epi', alt.Tooltip('temp_media:Q', format='.1f')]
            ).properties(height=220, title='Temperatura'), use_container_width=True)
        with col_s2:
            st.altair_chart(alt.Chart(seasonal).mark_bar(color='steelblue').encode(
                x=alt.X('semana_epi:O', title='Semana Epidemiológica'),
                y=alt.Y('precipitacao_total:Q', title='Precipitacao (mm)'),
                tooltip=['semana_epi', alt.Tooltip('precipitacao_total:Q', format='.1f')]
            ).properties(height=220, title='Precipitacao'), use_container_width=True)
        with col_s3:
            st.altair_chart(alt.Chart(seasonal).mark_line(color='teal').encode(
                x=alt.X('semana_epi:O', title='Semana Epidemiológica'),
                y=alt.Y('umidade_media:Q', title='Umidade (%)', scale=alt.Scale(domain=[0, 100])),
                tooltip=['semana_epi', alt.Tooltip('umidade_media:Q', format='.1f')]
            ).properties(height=220, title='Umidade'), use_container_width=True)


# ============================================================================
# TAB 5: SANEAMENTO
# ============================================================================

with tab5:
    st.header("Dados de Saneamento - SNIS & SINISA")

    # ---- SNIS ----
    st.subheader("SNIS (2020–2022)")
    st.caption(f"{snis['municipio'].nunique()} municipios | RS")

    anos_snis = st.multiselect(
        "Selecione os Anos (SNIS)",
        options=sorted(snis['year'].unique()),
        default=sorted(snis['year'].unique()),
        key="anos_snis"
    )
    filtered_snis = snis[snis['year'].isin(anos_snis)]

    if not filtered_snis.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Municípios", filtered_snis['municipio'].nunique())
        with col2:
            st.metric("Resíduos Coletados (ton/ano, media)", f"{filtered_snis['residuos_coletados'].mean():,.0f}")
        with col3:
            pct_coleta = filtered_snis['coleta_seletiva'].mean() * 100
            st.metric("Com Coleta Seletiva", f"{pct_coleta:.1f}%")

        st.markdown("**Resíduos Coletados por Ano**")
        residuos_ano = filtered_snis.groupby('year')['residuos_coletados'].sum().reset_index()
        st.altair_chart(
            alt.Chart(residuos_ano).mark_bar(color='seagreen').encode(
                x=alt.X('year:O', title='Ano'),
                y=alt.Y('residuos_coletados:Q', title='Resíduos Coletados (ton)'),
                tooltip=['year', alt.Tooltip('residuos_coletados:Q', format=',.0f')]
            ).properties(height=250),
            use_container_width=True
        )

        st.markdown("**Coleta Seletiva, Capinação e Catadores por Ano (%)**")
        bool_cols = ['coleta_seletiva', 'capinacao', 'catadores_dispersos']
        bool_pct = (
            filtered_snis.groupby('year')[bool_cols]
            .mean()
            .mul(100)
            .reset_index()
            .melt(id_vars='year', var_name='indicador', value_name='pct')
        )
        st.altair_chart(
            alt.Chart(bool_pct).mark_line(point=True).encode(
                x=alt.X('year:O', title='Ano'),
                y=alt.Y('pct:Q', title='Municípios com o serviço (%)', scale=alt.Scale(domain=[0, 100])),
                color=alt.Color('indicador:N', title='Indicador'),
                tooltip=['year', 'indicador', alt.Tooltip('pct:Q', format='.1f')]
            ).properties(height=250),
            use_container_width=True
        )

# ============================================================================
# TAB 4: QUALIDADE
# ============================================================================

with tab4:
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