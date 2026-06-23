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
    
    # TRATAMENTO DE NaN EM VARIÁVEIS DE DENGUE    
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
        
        # Coleta seletiva (booleano)
        'existencia_de_coleta_seletiva': 'coleta_seletiva',
        'tabela_cs01_informacoes_sobre_coleta_seletiva_de_residuos_solidos_existencia_de_coleta_seletiva': 'coleta_seletiva',
        'cs001_existe_coleta_seletiva_formalizada_pela_prefeitura_no_municipio': 'coleta_seletiva',
        
        # Varrição (numérico)
        'extensao_de_sarjeta_varrida': 'extensao_varricao',
        'extensao_de_sarjeta_varrida_total': 'extensao_varricao',
        'extensao_de_sarjetas_varridas': 'extensao_varricao',
        'tabela_va01_informacoes_sobre_servico_de_varricao_extensao_de_sarjeta_varrida': 'extensao_varricao',
        
        # Capinação (booleano)
        'servico_de_capina_e_rocada_existencia': 'capinacao',
        'cp001_existiu_o_servico_de_capina_e_rocada_no_municipio': 'capinacao',
        'tabela_cp01_informacoes_sobre_servicos_de_capina_e_rocada_servico_de_capina_e_rocada': 'capinacao',
        
        # Catadores (booleano)
        'existencia_de_catadores_dispersos': 'catadores_dispersos',
        'tabela_ca01_informacoes_sobre_catadores_existencia_de_catadores_dispersos': 'catadores_dispersos',
        'ca004_existem_catadores_de_materiais_reciclaveis_que_trabalham_dispersos_na_cidade': 'catadores_dispersos',
        
        # Quantidades (numérico - pode ter NaN)
        'quantidade_total_de_residuos_coletados_total': 'residuos_coletados',
        'quantidade_de_rdo_e_rpu_coletada_por_todos_os_agentes': 'residuos_coletados',
        'co119_quantidade_total_de_rdo_e_rpu_coletada_por_todos_os_agentes': 'residuos_coletados',
        
        # Trabalhadores (numérico - pode ter NaN)
        'quantidade_de_varredores_publico': 'varredores',
        'quantidade_de_varredores': 'varredores',
        'quantidade_de_varredores_1': 'varredores',
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
    # TRATAMENTO DE NaN PARA VARIÁVEIS NUMÉRICAS
    # ============================================================
    
    # 1. Para residuos_coletados: preencher com 0 (assumir que se não tem dado, é 0)
    #    Ou usar a mediana por ano/uf
    if 'residuos_coletados' in df.columns:
        # Converter para numérico
        df['residuos_coletados'] = pd.to_numeric(df['residuos_coletados'], errors='coerce')
        
        # Calcular mediana por ano e UF
        if 'uf' in df.columns:
            df['residuos_coletados'] = df.groupby(['year', 'uf'])['residuos_coletados'].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Preencher restantes com mediana global
        if df['residuos_coletados'].isna().any():
            global_median = df['residuos_coletados'].median()
            df['residuos_coletados'] = df['residuos_coletados'].fillna(global_median)
        
        # Se ainda houver NaN, preencher com 0
        df['residuos_coletados'] = df['residuos_coletados'].fillna(0)
    
    # 2. Para varredores: preencher com 0 (assumir que se não tem dado, é 0)
    if 'varredores' in df.columns:
        df['varredores'] = pd.to_numeric(df['varredores'], errors='coerce')
        
        # Calcular mediana por ano e UF
        if 'uf' in df.columns:
            df['varredores'] = df.groupby(['year', 'uf'])['varredores'].transform(
                lambda x: x.fillna(x.median())
            )
        
        # Preencher restantes com mediana global
        if df['varredores'].isna().any():
            global_median = df['varredores'].median()
            df['varredores'] = df['varredores'].fillna(global_median)
        
        # Se ainda houver NaN, preencher com 0
        df['varredores'] = df['varredores'].fillna(0)
    
    # 3. Para extensao_varricao: similar
    if 'extensao_varricao' in df.columns:
        df['extensao_varricao'] = pd.to_numeric(df['extensao_varricao'], errors='coerce')
        df['extensao_varricao'] = df['extensao_varricao'].fillna(0)
    
    # 4. Para variáveis booleanas: converter para boolean e preencher False
    bool_cols = ['coleta_seletiva', 'capinacao', 'catadores_dispersos']
    for col in bool_cols:
        if col in df.columns:
            # Converter para booleano
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.lower().map({'sim': True, 'não': False, 'nao': False})
            df[col] = df[col].fillna(False).astype(bool)
    
    # 5. Para populacao_total e populacao_urbana: usar mediana se faltar
    for col in ['populacao_total', 'populacao_urbana']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
    
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

def validate_transformation(df: pd.DataFrame, dataset_name: str) -> dict:
    """
    Valida os dados transformados.
    """
    validation = {
        'dataset': dataset_name,
        'shape': df.shape,
        'n_features': len(df.columns),
        'n_missing': df.isnull().sum().sum(),
        'columns_with_missing': df.columns[df.isnull().any()].tolist(),
        'missing_counts': df.isnull().sum().to_dict(),
        'date_range': None,
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    # Verificar colunas de data
    if 'ano' in df.columns and 'semana_epi' in df.columns:
        validation['date_range'] = f"{df['ano'].min()}-{df['semana_epi'].min()} a {df['ano'].max()}-{df['semana_epi'].max()}"
    
    # Verificar colunas específicas
    if 'outbreak' in df.columns:
        validation['target_distribution'] = df['outbreak'].value_counts().to_dict()
    
    return validation


def print_validation(validation: dict):
    """Imprime validação de forma legível."""
    print("\n" + "="*60)
    print(f"VALIDATION: {validation['dataset']}")
    print("="*60)
    print(f"Shape: {validation['shape']}")
    print(f"Features: {validation['n_features']}")
    print(f"Memory: {validation['memory_usage']:.2f} MB")
    
    if validation['date_range']:
        print(f"Date range: {validation['date_range']}")
    
    if validation['n_missing'] > 0:
        print(f"\n⚠️ Missing values: {validation['n_missing']}")
        print("Columns with missing:")
        for col in validation['columns_with_missing']:
            print(f"  {col}: {validation['missing_counts'][col]}")
    else:
        print("\n✅ No missing values found!")
    
    if 'target_distribution' in validation:
        print(f"\nTarget distribution (outbreak):")
        for label, count in validation['target_distribution'].items():
            print(f"  {label}: {count}")
    
    print("="*60)    

def create_unified_dataset(
    inmet_df: pd.DataFrame,
    dengue_df: pd.DataFrame,
    snis_df: pd.DataFrame = None,
    sinisa_df: pd.DataFrame = None,
    station_city_map: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Cria um dataset unificado combinando clima, dengue e saneamento.
    
    Args:
        inmet_df: Dados climáticos agregados por semana
        dengue_df: Dados de dengue preparados
        snis_df: Dados SNIS (2020-2022)
        sinisa_df: Dados SINISA (2023-2024)
        station_city_map: Mapeamento estação -> município (se disponível)
    """
    # 1. Merge clima + dengue
    unified = inmet_df.merge(
        dengue_df[['semana_id', 'casos', 'incidencia_100k', 'outbreak']],
        on='semana_id',
        how='left'
    )
    
    # 2. Adicionar dados de saneamento se disponíveis
    if snis_df is not None and not snis_df.empty:
        # Para cada ano, propagar features de saneamento para todas as semanas
        # Isso é uma simplificação - idealmente você teria dados mensais
        unified = unified.merge(
            snis_df[['ibge_code', 'year', 'populacao_total', 'coleta_seletiva', 'extensao_varricao']],
            left_on=['ano', 'estacao'],  # Ajuste conforme sua correspondência
            right_on=['year', 'ibge_code'],
            how='left'
        )
    
    # 3. Garantir consistência
    unified = normalize_column_names(unified)
    
    return unified    