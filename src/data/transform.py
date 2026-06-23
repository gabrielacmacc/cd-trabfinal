# src/data/transform.py

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from src.data.preprocess import (
    normalize_column_names,
    parse_dates,
)

# ---------------------------------------------------------------------------
# 3. DATA TRANSFORMATION - Estruturação dos Dados
# ---------------------------------------------------------------------------

# ----- 3.1 Temporal Structure ----------------------------------------------

def create_epidemiological_week(df: pd.DataFrame, date_col: str = 'data') -> pd.DataFrame:
    """
    Cria semana epidemiológica a partir de uma coluna de data.
    Padrão SINAN: semana epidemiológica começa no domingo.
    """
    df = df.copy()
    
    # Usar parse_dates do preprocess
    df['data'] = parse_dates(df[date_col])
    
    # ISO week (segunda a domingo) - ajustar para domingo a sábado (padrão SINAN)
    df['semana_epi'] = df['data'].dt.isocalendar().week
    df['ano_epi'] = df['data'].dt.isocalendar().year
    
    # Ajuste para semanas que cruzam anos
    mask_cross = (df['data'].dt.month == 1) & (df['semana_epi'] > 50)
    df.loc[mask_cross, 'ano_epi'] = df.loc[mask_cross, 'ano_epi'] - 1
    
    mask_cross2 = (df['data'].dt.month == 12) & (df['semana_epi'] == 1)
    df.loc[mask_cross2, 'ano_epi'] = df.loc[mask_cross2, 'ano_epi'] + 1
    
    # Criar chave única para merge
    df['semana_id'] = df['ano_epi'].astype(str) + '_SE' + df['semana_epi'].astype(str).str.zfill(2)
    
    return df


def extract_week_from_se(df: pd.DataFrame, se_col: str = 'se') -> pd.DataFrame:
    """
    Extrai ano e semana epidemiológica a partir do formato '202053'.
    Usado para dados do InfoDengue.
    """
    df = df.copy()
    df['ano'] = df[se_col].astype(str).str[:4].astype(int)
    df['semana_epi'] = df[se_col].astype(str).str[4:].astype(int)
    df['semana_id'] = df['ano'].astype(str) + '_SE' + df['semana_epi'].astype(str).str.zfill(2)
    return df


# ----- 3.2 Aggregation (INMET) ---------------------------------------------

def aggregate_inmet_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega dados horários do INMET para semana epidemiológica.
    """
    # Garantir que temos semana epidemiológica
    if 'semana_epi' not in df.columns:
        df = create_epidemiological_week(df)
    
    # Mapeamento de colunas do INMET para agregação
    agg_dict = {
        # Temperatura
        'temperatura_do_ar_bulbo_seco_horaria_c': ['mean', 'min', 'max', 'std'],
        'temperatura_maxima_na_hora_ant_aut_c': ['mean', 'max'],
        'temperatura_minima_na_hora_ant_aut_c': ['mean', 'min'],
        
        # Umidade
        'umidade_relativa_do_ar_horaria': ['mean', 'min', 'max', 'std'],
        
        # Precipitação
        'precipitacao_total_horario_mm': ['sum', 'mean', 'max'],
        
        # Vento
        'vento_velocidade_horaria_m_s': ['mean', 'max'],
        
        # Pressão (se existir)
        'pressao_atmosferica_ao_nivel_da_estacao_horaria_mb': ['mean', 'min', 'max'] 
            if 'pressao_atmosferica_ao_nivel_da_estacao_horaria_mb' in df.columns else None,
        
        # Rajada (se existir)
        'vento_rajada_maxima_m_s': ['mean', 'max'] 
            if 'vento_rajada_maxima_m_s' in df.columns else None,
        
        # Contagem de registros válidos
        'data': 'count'
    }
    
    # Remover None e filtrar colunas que existem
    agg_dict = {k: v for k, v in agg_dict.items() if v is not None and k in df.columns}
    
    # Agrupar por estação e semana
    grouped = df.groupby(['estacao', 'ano_epi', 'semana_epi', 'semana_id']).agg(agg_dict).reset_index()
    
    # Achatar colunas MultiIndex
    grouped.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in grouped.columns.values]
    
    # Renomear colunas para nomes limpos
    rename_map = {
        'estacao_': 'estacao',
        'ano_epi_': 'ano',
        'semana_epi_': 'semana_epi',
        'semana_id_': 'semana_id',
        'temperatura_do_ar_bulbo_seco_horaria_c_mean': 'temp_media',
        'temperatura_do_ar_bulbo_seco_horaria_c_min': 'temp_min',
        'temperatura_do_ar_bulbo_seco_horaria_c_max': 'temp_max',
        'temperatura_do_ar_bulbo_seco_horaria_c_std': 'temp_std',
        'temperatura_maxima_na_hora_ant_aut_c_mean': 'temp_max_media',
        'temperatura_maxima_na_hora_ant_aut_c_max': 'temp_max_abs',
        'temperatura_minima_na_hora_ant_aut_c_mean': 'temp_min_media',
        'temperatura_minima_na_hora_ant_aut_c_min': 'temp_min_abs',
        'umidade_relativa_do_ar_horaria_mean': 'umidade_media',
        'umidade_relativa_do_ar_horaria_min': 'umidade_min',
        'umidade_relativa_do_ar_horaria_max': 'umidade_max',
        'umidade_relativa_do_ar_horaria_std': 'umidade_std',
        'precipitacao_total_horario_mm_sum': 'precipitacao_total',
        'precipitacao_total_horario_mm_mean': 'precipitacao_media',
        'precipitacao_total_horario_mm_max': 'precipitacao_max',
        'vento_velocidade_horaria_m_s_mean': 'vento_medio',
        'vento_velocidade_horaria_m_s_max': 'vento_max',
        'vento_rajada_maxima_m_s_mean': 'rajada_media',
        'vento_rajada_maxima_m_s_max': 'rajada_max',
        'pressao_atmosferica_ao_nivel_da_estacao_horaria_mb_mean': 'pressao_media',
        'pressao_atmosferica_ao_nivel_da_estacao_horaria_mb_min': 'pressao_min',
        'pressao_atmosferica_ao_nivel_da_estacao_horaria_mb_max': 'pressao_max',
        'data_count': 'registros_semana'
    }
    
    # Aplicar renomeação apenas para colunas que existem
    for old, new in rename_map.items():
        if old in grouped.columns:
            grouped = grouped.rename(columns={old: new})
    
    # Garantir consistência com preprocess
    grouped = normalize_column_names(grouped)
    
    return grouped


# ----- 3.3 Prepare Dengue Data (Target) ------------------------------------

def prepare_dengue_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara dados de dengue para uso como target (variável alvo).
    Extrai semana epidemiológica e calcula incidência.
    
    Tratamento de NaN:
    - casos_est: mantém NaN (será usado como flag de incerteza)
    - casprov_est: preenchido com casos_est quando disponível
    - casconf: preenchido com casos_est (dados de confirmação)
    """
    df = df.copy()
    
    # Extrair ano e semana epidemiológica
    if 'se' in df.columns:
        df['ano'] = df['se'].astype(str).str[:4].astype(int)
        df['semana_epi'] = df['se'].astype(str).str[4:].astype(int)
        df['semana_id'] = df['ano'].astype(str) + '_SE' + df['semana_epi'].astype(str).str.zfill(2)
    
    # ============================================================
    # TRATAMENTO DE NaN EM VARIÁVEIS DE DENGUE
    # ============================================================
    
    # 1. casos_est: estimativa de casos (pode ter NaN)
    if 'casos_est' in df.columns:
        # Converter para numérico
        df['casos_est'] = pd.to_numeric(df['casos_est'], errors='coerce')
        # Não preencher - manter NaN como flag de baixa confiança
    
    # 2. casprov_est: casos prováveis estimados
    if 'casprov_est' in df.columns:
        df['casprov_est'] = pd.to_numeric(df['casprov_est'], errors='coerce')
        # Preencher com casos_est se disponível
        if 'casos_est' in df.columns:
            df['casprov_est'] = df['casprov_est'].fillna(df['casos_est'])
        # Se ainda NaN, preencher com 0
        df['casprov_est'] = df['casprov_est'].fillna(0)
    
    # 3. casprov_est_min e casprov_est_max
    for col in ['casprov_est_min', 'casprov_est_max']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Preencher com base em casprov_est
            if 'casprov_est' in df.columns:
                if col == 'casprov_est_min':
                    df[col] = df[col].fillna(df['casprov_est'] * 0.8)
                else:
                    df[col] = df[col].fillna(df['casprov_est'] * 1.2)
            df[col] = df[col].fillna(0)
    
    # 4. casconf: casos confirmados
    if 'casconf' in df.columns:
        df['casconf'] = pd.to_numeric(df['casconf'], errors='coerce')
        # Preencher com casos_est se disponível
        if 'casos_est' in df.columns:
            df['casconf'] = df['casconf'].fillna(df['casos_est'])
        # Se ainda NaN, preencher com 0
        df['casconf'] = df['casconf'].fillna(0)
    
    # 5. casos: número de casos (coluna principal)
    if 'casos' in df.columns:
        df['casos'] = pd.to_numeric(df['casos'], errors='coerce')
        # Preencher com casconf se disponível
        if 'casconf' in df.columns:
            df['casos'] = df['casos'].fillna(df['casconf'])
        # Preencher com casprov_est se disponível
        if 'casprov_est' in df.columns:
            df['casos'] = df['casos'].fillna(df['casprov_est'])
        # Se ainda NaN, preencher com 0
        df['casos'] = df['casos'].fillna(0)
    
    # 6. pop: população (necessário para incidência)
    if 'pop' in df.columns:
        df['pop'] = pd.to_numeric(df['pop'], errors='coerce')
        # Preencher com mediana por município/ano
        if 'localidade_id' in df.columns:
            df['pop'] = df.groupby('localidade_id')['pop'].transform(
                lambda x: x.fillna(x.median())
            )
        df['pop'] = df['pop'].fillna(df['pop'].median())
    
    # 7. Calcular incidência
    if 'p_inc100k' not in df.columns and 'casos' in df.columns and 'pop' in df.columns:
        # Evitar divisão por zero
        df['incidencia_100k'] = np.where(
            df['pop'] > 0,
            df['casos'] / (df['pop'] / 100000),
            0
        )
    elif 'p_inc100k' in df.columns:
        df['incidencia_100k'] = pd.to_numeric(df['p_inc100k'], errors='coerce')
        df['incidencia_100k'] = df['incidencia_100k'].fillna(0)
    
    # 8. Criar target: surto (≥20 casos/100k)
    if 'incidencia_100k' in df.columns:
        df['outbreak'] = (df['incidencia_100k'] >= 20).astype(int)
    
    # 9. Criar flags para variáveis com NaN
    for col in ['casos_est', 'casprov_est', 'casconf']:
        if col in df.columns:
            # Flag indicando se o dado original era NaN
            df[f'{col}_missing'] = df[col].isna().astype(int)
    
    # Garantir consistência com preprocess
    df = normalize_column_names(df)
    
    return df


# ----- 3.4 Prepare Sanitation Data (SNIS) ----------------------------------
def prepare_snis_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Prepara dados do SNIS (2020-2022) para uso no modelo.
    Extrai features relevantes de saneamento.
    """
    df = df.copy()
    df['year'] = year
    
    # Mapeamento de colunas comuns do SNIS
    rename_map = {
        # Identificação
        'codigo_do_municipio': 'ibge_code',
        'municipios_codigo_do_municipio': 'ibge_code',
        'codigo_do_municipio_1': 'ibge_code',
        'municipio': 'municipio',
        'municipios_municipio': 'municipio',
        'uf': 'uf',
        'municipios_uf': 'uf',
        'estado': 'uf',
        
        # População
        'total_populacao': 'populacao_total',
        'populacao_total': 'populacao_total',
        'tabela_ge01a_informacoes_gerais_total_populacao': 'populacao_total',
        'populacao_urbana': 'populacao_urbana',
        'tabela_ge01a_informacoes_gerais_populacao_urbana': 'populacao_urbana',
        
        # Coleta seletiva
        'existencia_de_coleta_seletiva': 'coleta_seletiva',
        'tabela_cs01_informacoes_sobre_coleta_seletiva_de_residuos_solidos_existencia_de_coleta_seletiva': 'coleta_seletiva',
        'cs001_existe_coleta_seletiva_formalizada_pela_prefeitura_no_municipio': 'coleta_seletiva',
        
        # Varrição
        'extensao_de_sarjeta_varrida': 'extensao_varricao',
        'extensao_de_sarjeta_varrida_total': 'extensao_varricao',
        'extensao_de_sarjetas_varridas': 'extensao_varricao',
        'tabela_va01_informacoes_sobre_servico_de_varricao_extensao_de_sarjeta_varrida': 'extensao_varricao',
        
        # Capinação
        'servico_de_capina_e_rocada_existencia': 'capinacao',
        'cp001_existiu_o_servico_de_capina_e_rocada_no_municipio': 'capinacao',
        'tabela_cp01_informacoes_sobre_servicos_de_capina_e_rocada_servico_de_capina_e_rocada': 'capinacao',
        
        # Catadores
        'existencia_de_catadores_dispersos': 'catadores_dispersos',
        'tabela_ca01_informacoes_sobre_catadores_existencia_de_catadores_dispersos': 'catadores_dispersos',
        'ca004_existem_catadores_de_materiais_reciclaveis_que_trabalham_dispersos_na_cidade': 'catadores_dispersos',
        
        # Quantidades
        'quantidade_total_de_residuos_coletados_total': 'residuos_coletados',
        'quantidade_de_rdo_e_rpu_coletada_por_todos_os_agentes': 'residuos_coletados',
        'co119_quantidade_total_de_rdo_e_rpu_coletada_por_todos_os_agentes': 'residuos_coletados',
        'quantidade_de_rdo_coletada_pelo_agente_publico': 'residuos_coletados',
        'co108_quantidade_de_rdo_coletada_pelo_agente_publico': 'residuos_coletados',
        
        # Trabalhadores
        'quantidade_de_varredores_publico': 'varredores',
        'quantidade_de_varredores': 'varredores',
        'quantidade_de_varredores_1': 'varredores',
        'quantidade_de_varredores_publico_1': 'varredores',
    }
    
    # Aplicar renomeação
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    
    # Selecionar colunas relevantes
    keep_cols = ['ibge_code', 'municipio', 'uf', 'year', 
                 'populacao_total', 'populacao_urbana',
                 'coleta_seletiva', 'extensao_varricao', 'capinacao',
                 'catadores_dispersos', 'residuos_coletados', 'varredores']
    
    # Manter apenas colunas que existem
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    
    # ============================================================
    # TRATAMENTO ROBUSTO DE NaN
    # ============================================================
    
    # 1. Garantir que colunas numéricas são float
    numeric_cols = ['populacao_total', 'populacao_urbana', 'residuos_coletados', 
                    'varredores', 'extensao_varricao']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 2. Tratar residuos_coletados
    if 'residuos_coletados' in df.columns:
        # Criar flag de missing
        df['residuos_coletados_missing'] = df['residuos_coletados'].isna().astype(int)
        
        # Método 1: Preencher com mediana por (UF, ano)
        if 'uf' in df.columns and 'year' in df.columns:
            df['residuos_coletados'] = df.groupby(['uf', 'year'])['residuos_coletados'].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Método 2: Preencher com mediana por ano
        if df['residuos_coletados'].isna().any() and 'year' in df.columns:
            df['residuos_coletados'] = df.groupby('year')['residuos_coletados'].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Método 3: Preencher com mediana global
        if df['residuos_coletados'].isna().any():
            global_median = df['residuos_coletados'].median()
            df['residuos_coletados'] = df['residuos_coletados'].fillna(global_median)
        
        # Método 4: Se ainda houver NaN, preencher com 0
        df['residuos_coletados'] = df['residuos_coletados'].fillna(0)
    
    # 3. Tratar varredores
    if 'varredores' in df.columns:
        # Criar flag de missing
        df['varredores_missing'] = df['varredores'].isna().astype(int)
        
        # Método 1: Preencher com mediana por (UF, ano)
        if 'uf' in df.columns and 'year' in df.columns:
            df['varredores'] = df.groupby(['uf', 'year'])['varredores'].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Método 2: Preencher com mediana por ano
        if df['varredores'].isna().any() and 'year' in df.columns:
            df['varredores'] = df.groupby('year')['varredores'].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Método 3: Preencher com mediana global
        if df['varredores'].isna().any():
            global_median = df['varredores'].median()
            df['varredores'] = df['varredores'].fillna(global_median)
        
        # Método 4: Se ainda houver NaN, preencher com 0
        df['varredores'] = df['varredores'].fillna(0)
    
    # 4. Tratar extensao_varricao
    if 'extensao_varricao' in df.columns:
        # Criar flag de missing
        df['extensao_varricao_missing'] = df['extensao_varricao'].isna().astype(int)
        
        # Preencher com mediana por (UF, ano)
        if 'uf' in df.columns and 'year' in df.columns:
            df['extensao_varricao'] = df.groupby(['uf', 'year'])['extensao_varricao'].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Preencher com mediana global
        if df['extensao_varricao'].isna().any():
            df['extensao_varricao'] = df['extensao_varricao'].fillna(
                df['extensao_varricao'].median()
            )
        
        # Se ainda houver NaN, preencher com 0
        df['extensao_varricao'] = df['extensao_varricao'].fillna(0)
    
    # 5. Tratar populução
    for col in ['populacao_total', 'populacao_urbana']:
        if col in df.columns:
            # Preencher com mediana por ano
            if 'year' in df.columns:
                df[col] = df.groupby('year')[col].transform(
                    lambda x: x.fillna(x.median())
                )
            # Preencher com mediana global
            df[col] = df[col].fillna(df[col].median())
            # Se ainda houver NaN, preencher com 0
            df[col] = df[col].fillna(0)
    
    # 6. Tratar variáveis booleanas
    bool_cols = ['coleta_seletiva', 'capinacao', 'catadores_dispersos']
    for col in bool_cols:
        if col in df.columns:
            # Converter strings para boolean
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.lower().map(
                    {'sim': True, 'não': False, 'nao': False, 'true': True, 'false': False}
                )
            # Preencher NaN com False
            df[col] = df[col].fillna(False).astype(bool)
    
    # Garantir consistência com preprocess
    df = normalize_column_names(df)
    
    return df


def combine_snis_data(snis_2020: pd.DataFrame, 
                      snis_2021: pd.DataFrame, 
                      snis_2022: pd.DataFrame) -> pd.DataFrame:
    """
    Combina dados do SNIS de 2020, 2021 e 2022 em um único DataFrame.
    """
    # Preparar cada ano
    snis_list = []
    
    if snis_2020 is not None and not snis_2020.empty:
        snis_list.append(prepare_snis_data(snis_2020, 2020))
    
    if snis_2021 is not None and not snis_2021.empty:
        snis_list.append(prepare_snis_data(snis_2021, 2021))
    
    if snis_2022 is not None and not snis_2022.empty:
        snis_list.append(prepare_snis_data(snis_2022, 2022))
    
    if not snis_list:
        return pd.DataFrame()
    
    # Combinar
    combined = pd.concat(snis_list, ignore_index=True)
    
    # Garantir consistência
    combined = normalize_column_names(combined)
    
    return combined


# ----- 3.5 Prepare Sanitation Data (SINISA) --------------------------------

def prepare_sinisa_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara dados do SINISA (2023-2024) para uso no modelo.
    """
    df = df.copy()
    
    # Mapeamento de colunas do SINISA
    rename_map = {
        'codigo_do_ibge': 'ibge_code',
        'municipio': 'municipio',
        'uf': 'uf',
        'populacao_total': 'populacao_total',
        'quantidade_de_domicilios_rurais_existente_no_municipio': 'domicilios_rurais',
        'area_km2': 'area_km2',
        'executor_do_servico_de_varricao_de_sarjetas_e_logradouros_publicos': 'executor_varricao',
        'extensao_de_sarjetas_varridas': 'extensao_varricao',
        'ha_algum_tipo_de_varricao_mecanizada_no_municipio': 'varricao_mecanizada',
        'houve_contratacao_de_trabalhadores_temporarios_para_reforco_na_execucao_de_algum_dos_servicos_de_limpeza_urbana_e_ou_manejo_de_residuos_solidos': 'trabalhadores_temporarios',
        'trabalhadores_temporarios_contratados': 'qtd_trabalhadores_temporarios',
    }
    
    # Aplicar renomeação
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    
    # Extrair ano da coluna 'year' ou de 'ano_referencia'
    if 'year' not in df.columns and 'ano_de_referencia' in df.columns:
        df['year'] = df['ano_de_referencia']
    
    # Selecionar colunas relevantes
    keep_cols = ['ibge_code', 'municipio', 'uf', 'year',
                 'populacao_total', 'domicilios_rurais', 'area_km2',
                 'executor_varricao', 'extensao_varricao', 'varricao_mecanizada',
                 'trabalhadores_temporarios', 'qtd_trabalhadores_temporarios']
    
    # Manter apenas colunas que existem
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    
    # Garantir consistência com preprocess
    df = normalize_column_names(df)
    
    return df


# ----- 3.6 Sanitation Feature Extraction -----------------------------------

def extract_sanitation_features_for_merge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai features de saneamento para merge com dados climáticos.
    """
    df = df.copy()
    
    # Garantir que ibge_code é string com 6 dígitos (padrão)
    if 'ibge_code' in df.columns:
        df['ibge_code'] = df['ibge_code'].astype(str).str.zfill(6)
    
    return df


# ----- 3.7 Save/Load Functions ---------------------------------------------

def save_transformed_data(df: pd.DataFrame, name: str, output_dir: str = 'data/processed/'):
    """
    Salva dados transformados em Parquet.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Salvar
    df.to_parquet(f'{output_dir}/{name}.parquet', index=False)
    print(f"✓ Dados salvos em: {output_dir}{name}.parquet")
    
    # Salvar metadados
    import json
    metadata = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'transformation_date': pd.Timestamp.now().isoformat(),
        'n_features': len(df.columns)
    }
    
    with open(f'{output_dir}/{name}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return df


def load_transformed_data(name: str, input_dir: str = 'data/processed/') -> pd.DataFrame:
    """
    Carrega dados transformados do Parquet.
    """
    return pd.read_parquet(f'{input_dir}/{name}.parquet')