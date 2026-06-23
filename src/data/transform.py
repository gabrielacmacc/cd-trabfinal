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

# ----- 3.4 Prepare Sanitation Data (SNIS + SINISA Consolidado) -------------

# src/data/transform.py - Função para extrair features de saneamento

def extract_sanitation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai e cria features relevantes de saneamento para previsão de dengue.
    """
    df = df.copy()
    
    # ============================================================
    # 1. FEATURES ESSENCIAIS 
    # ============================================================
    
    keep_cols = [
        # Identificadores
        'codigo_do_ibge_cod_ibge',
        'ano_de_referencia',
        'uf',
        'municipio_nom_mun',
        
        # População
        'populacao_total_dfe0001',
        'populacao_urbana_dfe0002',
        'populacao_rural_dfe0003',
        
        # Área
        'area_km2_ogm0005',
        
        # Domicílios
        'quantidade_de_domicilios_totais_existente_no_municipio_ogm4006',
        
        # Cobertura de Resíduos (CRÍTICO)
        'irs0001_cobertura_da_populacao_total_com_coleta_de_residuos_solidos_domiciliares_percentual',
        'irs0005_cobertura_da_populacao_total_com_coleta_seletiva_de_residuos_solidos_domiciliares_percentual',
        
        # Coleta Seletiva (CRÍTICO)
        'irs0006_cobertura_da_populacao_urbana_com_coleta_seletiva_direta_de_residuos_solidos_domiciliares_percentual',
        
        # Flag de resposta
        'respondeu_ao_modulo_de_manejo_de_residuos_solidos_2024_sim_nao'
    ]
    
    # Manter apenas colunas que existem
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
        
    rename_map = {
        'codigo_do_ibge_cod_ibge': 'ibge_code',
        'ano_de_referencia': 'year',
        'municipio_nom_mun': 'municipio',
        'uf': 'uf',
        'populacao_total_dfe0001': 'populacao_total',
        'populacao_urbana_dfe0002': 'populacao_urbana',
        'populacao_rural_dfe0003': 'populacao_rural',
        'area_km2_ogm0005': 'area_km2',
        'quantidade_de_domicilios_totais_existente_no_municipio_ogm4006': 'domicilios_totais',
        'irs0001_cobertura_da_populacao_total_com_coleta_de_residuos_solidos_domiciliares_percentual': 'cobertura_total',
        'irs0005_cobertura_da_populacao_total_com_coleta_seletiva_de_residuos_solidos_domiciliares_percentual': 'coleta_seletiva_total',
        'irs0006_cobertura_da_populacao_urbana_com_coleta_seletiva_direta_de_residuos_solidos_domiciliares_percentual': 'coleta_seletiva_urbana',
        'respondeu_ao_modulo_de_manejo_de_residuos_solidos_2024_sim_nao': 'respondeu_modulo'
    }
    
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    
    # ============================================================
    # 3. FEATURES DERIVADAS (CRÍTICAS PARA DENGUE)
    # ============================================================
    
    # 3.1 Densidade populacional
    if 'populacao_total' in df.columns and 'area_km2' in df.columns:
        df['densidade_populacional'] = df['populacao_total'] / df['area_km2']
        df['densidade_populacional'] = df['densidade_populacional'].replace([np.inf, -np.inf], np.nan)
    
    # 3.2 Proporção urbana
    if 'populacao_urbana' in df.columns and 'populacao_total' in df.columns:
        df['proporcao_urbana'] = df['populacao_urbana'] / df['populacao_total']
        df['proporcao_rural'] = 1 - df['proporcao_urbana']
    
    # 3.3 Lacuna de coleta seletiva (MUITO IMPORTANTE para dengue!)
    if 'cobertura_total' in df.columns and 'coleta_seletiva_total' in df.columns:
        df['lacuna_coleta_seletiva'] = df['cobertura_total'] - df['coleta_seletiva_total']
        df['lacuna_coleta_seletiva'] = df['lacuna_coleta_seletiva'].clip(lower=0)
    
    # 3.4 Qualidade do serviço
    if 'coleta_seletiva_total' in df.columns and 'cobertura_total' in df.columns:
        df['qualidade_servico'] = (df['coleta_seletiva_total'] / (df['cobertura_total'] + 1)) * 100
        df['qualidade_servico'] = df['qualidade_servico'].clip(upper=100)
    
    # 3.5 Índice de Risco de Saneamento (quanto maior, pior)
    risk_components = []
    if 'lacuna_coleta_seletiva' in df.columns:
        risk_components.append(df['lacuna_coleta_seletiva'] / 100)
    if 'cobertura_total' in df.columns:
        risk_components.append(1 - df['cobertura_total'] / 100)
    
    if risk_components:
        df['risco_saneamento'] = np.mean(risk_components, axis=0) * 100
    
    # 3.6 Cobertura efetiva (ajustada pela urbanização)
    if 'cobertura_total' in df.columns and 'proporcao_urbana' in df.columns:
        df['cobertura_efetiva'] = df['cobertura_total'] * (0.7 + 0.3 * df['proporcao_urbana'])
    
    # 3.7 Flag: cobertura alta (>80%)
    if 'cobertura_total' in df.columns:
        df['cobertura_alta'] = (df['cobertura_total'] > 80).astype(int)
        df['cobertura_baixa'] = (df['cobertura_total'] < 50).astype(int)
    
    # 3.8 Flag: tem coleta seletiva
    if 'coleta_seletiva_total' in df.columns:
        df['tem_coleta_seletiva'] = (df['coleta_seletiva_total'] > 0).astype(int)
        df['coleta_seletiva_alta'] = (df['coleta_seletiva_total'] > 50).astype(int)
    
    # 3.9 Flag: densidade alta
    if 'densidade_populacional' in df.columns:
        df['densidade_alta'] = (df['densidade_populacional'] > 100).astype(int)
    
    # 3.10 Índice de Desenvolvimento do Saneamento (IDS)
    if all(c in df.columns for c in ['cobertura_total', 'coleta_seletiva_total', 'qualidade_servico']):
        cobertura_norm = df['cobertura_total'] / 100
        seletiva_norm = df['coleta_seletiva_total'] / 100
        qualidade_norm = df['qualidade_servico'] / 100
        df['ids'] = (cobertura_norm * 0.4 + seletiva_norm * 0.35 + qualidade_norm * 0.25) * 100
    
    # ============================================================
    # 4. TRATAR VALORES FALTANTES
    # ============================================================
    
    # Preencher com 0 onde não há dados
    fill_zero_cols = [
        'cobertura_total', 'coleta_seletiva_total', 'coleta_seletiva_urbana',
        'lacuna_coleta_seletiva', 'qualidade_servico', 'risco_saneamento'
    ]
    
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # ============================================================
    # 5. SELECIONAR APENAS FEATURES RELEVANTES
    # ============================================================
    
    final_cols = [
        # Identificadores
        'ibge_code', 'year', 'uf', 'municipio',
        
        # Features principais
        'populacao_total', 'populacao_urbana', 'area_km2',
        'densidade_populacional', 'proporcao_urbana',
        
        # Features de cobertura (CRÍTICAS)
        'cobertura_total', 'coleta_seletiva_total',
        'lacuna_coleta_seletiva', 'qualidade_servico',
        'risco_saneamento', 'cobertura_efetiva',
        
        # Flags
        'cobertura_alta', 'cobertura_baixa',
        'tem_coleta_seletiva', 'coleta_seletiva_alta',
        'densidade_alta',
        
        # Índices
        'ids'
    ]
    
    # Manter apenas colunas que existem
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].copy()
    
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


def unify_datasets(
    climate_df: pd.DataFrame,
    dengue_df: pd.DataFrame,
    sanitation_df: pd.DataFrame,
    station_city_map: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Une datasets e mantém target junto para análise.
    
    O dataset retornado contém target (outbreak) junto com features,
    mas APENAS para análise. Na modelagem, separe X e y.
    
    Returns:
        DataFrame unificado com target incluído
    """
    
    print("\n=== UNIFICANDO DATASETS (Com target incluso) ===")
    
    # 1. Selecionar features climáticas (X)
    climate_features = climate_df.copy()
    if climate_features['semana_id'].duplicated().any():
        print("   Agrupando linhas com o mesmo semana_id...")
        agg_columns = {}
        for col in climate_features.columns:
            if col == 'semana_id':
                continue
            if pd.api.types.is_numeric_dtype(climate_features[col]):
                agg_columns[col] = 'mean'
            else:
                agg_columns[col] = 'first'
        climate_features = climate_features.groupby('semana_id', as_index=False).agg(agg_columns)
        print(f"   Após agregação: {len(climate_features)} linhas")
    print(f"   Features climáticas: {len(climate_features.columns)}")
    
    # 2. Selecionar target (y)
    dengue_target = dengue_df[['semana_id', 'outbreak']].copy()
    # Manter casos para referência (NUNCA usar como feature!)
    if 'casos' in dengue_df.columns:
        dengue_target['casos'] = dengue_df['casos']
    
    print(f"   Target: {dengue_target['outbreak'].sum()} surtos ({dengue_target['outbreak'].mean()*100:.1f}%)")
    
    # 3. Merge (target fica junto!)
    unified = climate_features.merge(
        dengue_target,
        on='semana_id',
        how='left'
    )
    
    # 4. Adicionar saneamento
    if sanitation_df is not None and station_city_map is not None:
        # Mapear estações para municípios
        unified = unified.merge(station_city_map, on='estacao', how='left')
        
        # Merge com saneamento
        sanitation_features = sanitation_df[[
            'ibge_code', 'year', 'cobertura_total', 'coleta_seletiva_total',
            'densidade_populacional', 'proporcao_urbana', 'qualidade_servico'
        ]].copy()
        
        unified = unified.merge(
            sanitation_features,
            left_on=['ibge_code', 'ano'],
            right_on=['ibge_code', 'year'],
            how='left'
        )
        
        # Remover colunas duplicadas
        if 'year' in unified.columns:
            unified = unified.drop(columns=['year'])
    
    # 5. Remover duplicatas
    before_dedup = len(unified)
    unified = unified.drop_duplicates(subset=['semana_id'], keep='first')
    print(f"   Removidas {before_dedup - len(unified)} linhas duplicadas")
    
    # 6. Remover linhas sem target
    before = len(unified)
    unified = unified.dropna(subset=['outbreak'])
    print(f"   Removidas {before - len(unified)} linhas sem target")
    
    print(f"   Dataset final: {unified.shape}")
    print(f"   Features: {len(unified.columns)} colunas (incluindo target)")
    
    return unified