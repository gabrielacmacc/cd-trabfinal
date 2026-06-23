
import pandas as pd
import numpy as np
from typing import List

# ---------------------------------------------------------------------------
# CLIMATE FEATURES
# ---------------------------------------------------------------------------

def create_climate_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features baseadas em limiares climáticos críticos para dengue.
    """
    df = df.copy()
    
    # 1. Temperatura mínima crítica (transmissão ativa)
    df['temp_min_acima_18'] = (df['temp_min'] > 18).astype(int)
    
    # 2. Semanas consecutivas com temp > 18°C
    def consecutive_weeks(series):
        result = []
        count = 0
        for val in series:
            if val == 1:
                count += 1
            else:
                count = 0
            result.append(count)
        return result
    
    df['semanas_consecutivas_18'] = df.groupby('estacao')['temp_min_acima_18'].transform(
        lambda x: consecutive_weeks(x.tolist())
    )
    
    # 3. Limiares de umidade
    df['umidade_alta'] = (df['umidade_media'] > 75).astype(int)
    df['umidade_ideal'] = ((df['umidade_media'] >= 60) & (df['umidade_media'] <= 75)).astype(int)
    df['umidade_baixa'] = (df['umidade_media'] < 40).astype(int)
    
    # 4. Amplitude térmica
    df['amplitude_termica'] = df['temp_max'] - df['temp_min']
    
    # 5. Chuva moderada (favorável para criadouros)
    df['chuva_moderada'] = ((df['precipitacao_total'] >= 50) & (df['precipitacao_total'] <= 150)).astype(int)
    
    return df


def create_dengue_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria índices compostos para predição de dengue.
    Baseado em: Wallau et al., 2025
    """
    df = df.copy()
    
    # 1. Índice de Adequação (Index P)
    def calc_index_p(row):
        T = row['temp_media']
        UR = row['umidade_media']
        P = row['precipitacao_total']
        
        # Temperatura ideal: 22-28°C
        T_score = np.exp(-((T - 25)**2) / (2 * 5**2))
        
        # Umidade ideal: 60-75%
        if UR < 60:
            UR_score = np.exp(-((UR - 60)**2) / (2 * 10**2))
        elif UR > 75:
            UR_score = np.exp(-((UR - 75)**2) / (2 * 10**2))
        else:
            UR_score = 1.0
        
        # Precipitação: 50-150mm/semana é favorável
        if P < 50:
            P_score = np.exp(-((P - 50)**2) / (2 * 20**2))
        elif P > 150:
            P_score = np.exp(-((P - 150)**2) / (2 * 30**2))
        else:
            P_score = 1.0
        
        return T_score * UR_score * P_score
    
    df['index_p'] = df.apply(calc_index_p, axis=1)
    
    # 2. Índice de Transmissibilidade
    df['transmissibilidade'] = np.where(
        df['temp_min'] > 18,
        (df['temp_min'] - 18) / (30 - 18),
        0
    ).clip(0, 1)
    
    # 3. Índice de Risco Climático (combinação simplificada)
    df['risco_climatico'] = (
        df['index_p'] * 0.6 + 
        df['transmissibilidade'] * 0.4
    )
    
    return df


# ---------------------------------------------------------------------------
# LAG FEATURES
# ---------------------------------------------------------------------------

def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int] = [1, 2, 3, 4, 6, 8],
    group_col: str = 'estacao'
) -> pd.DataFrame:
    """
    Cria features defasadas (lags) para capturar latência biológica.
    """
    df = df.copy()
    df = df.sort_values([group_col, 'ano', 'semana_epi'])
    
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            lag_col = f'{col}_lag_{lag}'
            df[lag_col] = df.groupby(group_col)[col].shift(lag)
    
    return df


# ---------------------------------------------------------------------------
# ROLLING WINDOW FEATURES
# ---------------------------------------------------------------------------

def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [2, 4, 8],
    group_col: str = 'estacao'
) -> pd.DataFrame:
    """
    Cria médias móveis, tendências e acumulados.
    """
    df = df.copy()
    df = df.sort_values([group_col, 'ano', 'semana_epi'])
    
    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            # Média móvel
            ma_col = f'{col}_ma_{window}'
            df[ma_col] = df.groupby(group_col)[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Tendência (variação em relação à média)
            trend_col = f'{col}_tendencia_{window}'
            df[trend_col] = df[col] - df[ma_col]
            
            # Variabilidade (desvio padrão)
            std_col = f'{col}_std_{window}'
            df[std_col] = df.groupby(group_col)[col].transform(
                lambda x: x.rolling(window, min_periods=2).std()
            )
    
    # Acumulado de precipitação
    if 'precipitacao_total' in df.columns:
        for window in windows:
            df[f'precipitacao_acum_{window}'] = df.groupby(group_col)['precipitacao_total'].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )
    
    return df


# ---------------------------------------------------------------------------
# EXTREME EVENTS
# ---------------------------------------------------------------------------

def create_extreme_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica eventos climáticos extremos.
    """
    df = df.copy()
    df = df.sort_values(['estacao', 'ano', 'semana_epi'])
    
    def mark_consecutive(series, min_consecutive=3):
        result = []
        count = 0
        for val in series:
            if val == 1:
                count += 1
            else:
                count = 0
            result.append(1 if count >= min_consecutive else 0)
        return result
    
    # 1. Onda de calor (temp_max > 30°C por 3 semanas)
    if 'temp_max' in df.columns:
        df['temp_max_30'] = (df['temp_max'] > 30).astype(int)
        df['onda_calor'] = df.groupby('estacao')['temp_max_30'].transform(
            lambda x: mark_consecutive(x.tolist(), 3)
        )
    
    # 2. Seca prolongada (precipitacao < 10mm por 3 semanas)
    if 'precipitacao_total' in df.columns:
        df['seca_semana'] = (df['precipitacao_total'] < 10).astype(int)
        df['seca_prolongada'] = df.groupby('estacao')['seca_semana'].transform(
            lambda x: mark_consecutive(x.tolist(), 3)
        )
    
    # 3. Chuva intensa e extrema
    if 'precipitacao_total' in df.columns:
        df['chuva_intensa'] = (df['precipitacao_total'] > 100).astype(int)
        df['chuva_extrema'] = (df['precipitacao_total'] > 150).astype(int)
        df['chuva_muito_baixa'] = (df['precipitacao_total'] < 5).astype(int)
    
    # 4. Oscilação térmica brusca
    if 'amplitude_termica' in df.columns:
        df['oscilacao_termica_brusca'] = (df['amplitude_termica'] > 10).astype(int)
    
    # 5. Umidade crítica
    if 'umidade_media' in df.columns:
        df['umidade_critica_baixa'] = (df['umidade_media'] < 40).astype(int)
        df['umidade_critica_alta'] = (df['umidade_media'] > 85).astype(int)
    
    return df


# ---------------------------------------------------------------------------
# TEMPORAL FEATURES
# ---------------------------------------------------------------------------

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features de sazonalidade e ciclos temporais.
    """
    df = df.copy()
    
    # Criar coluna de data se não existir
    if 'data' not in df.columns and 'ano' in df.columns and 'semana_epi' in df.columns:
        df['data'] = pd.to_datetime(
            df['ano'].astype(str) + '-01-01'
        ) + pd.to_timedelta((df['semana_epi'] - 1) * 7, unit='D')
    
    if 'data' in df.columns:
        df['mes'] = df['data'].dt.month
        df['trimestre'] = df['data'].dt.quarter
        
        # Período de risco (março a maio - pico sazonal no RS)
        df['periodo_risco'] = df['mes'].isin([3, 4, 5]).astype(int)
        df['verao'] = df['mes'].isin([12, 1, 2]).astype(int)
        df['outono'] = df['mes'].isin([3, 4, 5]).astype(int)
        df['inverno'] = df['mes'].isin([6, 7, 8]).astype(int)
        df['primavera'] = df['mes'].isin([9, 10, 11]).astype(int)
        
        # Features cíclicas (seno/cosseno) para capturar sazonalidade
        df['semana_sin'] = np.sin(2 * np.pi * df['semana_epi'] / 52)
        df['semana_cos'] = np.cos(2 * np.pi * df['semana_epi'] / 52)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
    
    # Ano como feature (tendência linear)
    if 'ano' in df.columns:
        df['ano_norm'] = (df['ano'] - df['ano'].min()) / (df['ano'].max() - df['ano'].min() + 1)
    
    return df


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING PIPELINE
# ---------------------------------------------------------------------------

def feature_engineering_pipeline(
    df: pd.DataFrame,
    create_lags: bool = True,
    create_rolling: bool = True,
    create_extremes: bool = True,
    create_temporal: bool = True
) -> pd.DataFrame:
    """
    Pipeline completo de Feature Engineering.
    """
    print("1. Criando limiares climáticos...")
    df = create_climate_thresholds(df)
    
    print("2. Criando índices de dengue...")
    df = create_dengue_indices(df)
    
    if create_lags:
        print("3. Criando features defasadas (lags)...")
        lag_cols = ['temp_media', 'temp_min', 'temp_max', 'umidade_media', 'precipitacao_total', 'index_p']
        lag_cols = [c for c in lag_cols if c in df.columns]
        df = create_lag_features(df, lag_cols, lags=[1, 2, 3, 4, 6, 8])
    
    if create_rolling:
        print("4. Criando médias móveis e acumulados...")
        rolling_cols = ['temp_media', 'temp_min', 'umidade_media', 'index_p']
        rolling_cols = [c for c in rolling_cols if c in df.columns]
        df = create_rolling_features(df, rolling_cols, windows=[2, 4, 8])
    
    if create_extremes:
        print("5. Identificando eventos climáticos extremos...")
        df = create_extreme_events(df)
    
    if create_temporal:
        print("6. Criando features temporais e sazonais...")
        df = create_temporal_features(df)
    
    print(f"✓ Feature Engineering concluído. Shape final: {df.shape}")
    
    return df


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING PIPELINE
# ---------------------------------------------------------------------------

def feature_engineering_pipeline(
    df: pd.DataFrame,
    create_lags: bool = True,
    create_rolling: bool = True,
    create_extremes: bool = True,
    create_temporal: bool = True
) -> pd.DataFrame:
    """
    Pipeline completo de Feature Engineering.
    """
    print("1. Criando limiares climáticos...")
    df = create_climate_thresholds(df)
    
    print("2. Criando índices de dengue...")
    df = create_dengue_indices(df)
    
    if create_lags:
        print("3. Criando features defasadas (lags)...")
        lag_cols = ['temp_media', 'temp_min', 'temp_max', 'umidade_media', 'precipitacao_total', 'index_p']
        lag_cols = [c for c in lag_cols if c in df.columns]
        df = create_lag_features(df, lag_cols, lags=[1, 2, 3, 4, 6, 8])
    
    if create_rolling:
        print("4. Criando médias móveis e acumulados...")
        rolling_cols = ['temp_media', 'temp_min', 'umidade_media', 'index_p']
        rolling_cols = [c for c in rolling_cols if c in df.columns]
        df = create_rolling_features(df, rolling_cols, windows=[2, 4, 8])
    
    if create_extremes:
        print("5. Identificando eventos climáticos extremos...")
        df = create_extreme_events(df)
    
    if create_temporal:
        print("6. Criando features temporais e sazonais...")
        df = create_temporal_features(df)
    
    print(f"✓ Feature Engineering concluído. Shape final: {df.shape}")
    
    return df


# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------

def validate_features(df: pd.DataFrame) -> dict:
    """
    Valida as features criadas.
    """
    validation = {
        'shape': df.shape,
        'n_features': len(df.columns),
        'n_missing': df.isnull().sum().sum(),
        'features': {
            'thresholds': [c for c in df.columns if c in ['temp_min_acima_18', 'semanas_consecutivas_18', 'umidade_ideal']],
            'indices': [c for c in df.columns if c in ['index_p', 'transmissibilidade', 'risco_climatico']],
            'lags': [c for c in df.columns if '_lag_' in c],
            'rolling': [c for c in df.columns if '_ma_' in c or '_acum_' in c],
            'extremos': [c for c in df.columns if c in ['onda_calor', 'chuva_intensa', 'seca_prolongada']],
            'temporais': [c for c in df.columns if c in ['semana_sin', 'semana_cos', 'periodo_risco']]
        }
    }
    return validation


def print_feature_summary(validation: dict):
    """Imprime sumário das features criadas."""
    print("\n" + "="*50)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*50)
    print(f"Shape:              {validation['shape']}")
    print(f"Total Features:     {validation['n_features']}")
    print(f"Missing Values:     {validation['n_missing']}")
    print("\n--- FEATURES BY CATEGORY ---")
    
    category_names = {
        'thresholds': 'Limiares Climáticos',
        'indices': 'Índices de Dengue',
        'lags': 'Defasagens (Lags)',
        'rolling': 'Médias Móveis',
        'extremos': 'Eventos Extremos',
        'temporais': 'Temporais/Sazonais'
    }
    
    for key, name in category_names.items():
        features = validation['features'].get(key, [])
        if features:
            print(f"\n{name}: {len(features)} features")
            if len(features) <= 5:
                print(f"  {features}")
            else:
                print(f"  {features[:3]} ... +{len(features)-3} more")
    print("="*50)

# ============================================================================
# SANITATION FEATURES (SNIS + SINISA)
# ============================================================================

def create_sanitation_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas a partir dos dados de saneamento (SNIS/SINISA).
    
    Features criadas:
    - densidade_populacional: População / Área
    - proporcao_urbana: População urbana / População total
    - proporcao_rural: 1 - proporcao_urbana
    - moradores_por_domicilio: População / Domicílios totais
    - coleta_seletiva_rural: Coleta seletiva total - Coleta seletiva urbana
    - cobertura_rural_estimada: Cobertura total - Cobertura urbana
    - lacuna_coleta_seletiva: Cobertura total - Coleta seletiva total
    - qualidade_servico: (Coleta seletiva / Cobertura total) * 100
    - eficiencia_coleta: Razão entre coleta direta e cobertura total
    - cobertura_media: Média das coberturas
    - infraestrutura_score: Score combinado de infraestrutura
    """
    df = df.copy()
    
    # 1. Densidade populacional
    if 'populacao_total' in df.columns and 'area_km2' in df.columns:
        df['densidade_populacional'] = df['populacao_total'] / df['area_km2']
        df['densidade_populacional'] = df['densidade_populacional'].replace([np.inf, -np.inf], np.nan)
    
    # 2. Proporção urbana/rural
    if 'populacao_urbana' in df.columns and 'populacao_total' in df.columns:
        df['proporcao_urbana'] = df['populacao_urbana'] / df['populacao_total']
        df['proporcao_rural'] = 1 - df['proporcao_urbana']
    
    # 3. Moradores por domicílio
    if 'populacao_total' in df.columns and 'domicilios_totais' in df.columns:
        df['moradores_por_domicilio'] = df['populacao_total'] / df['domicilios_totais']
        df['moradores_por_domicilio'] = df['moradores_por_domicilio'].replace([np.inf, -np.inf], np.nan)
    
    # 4. Coleta seletiva rural (estimada)
    if 'coleta_seletiva_total' in df.columns and 'coleta_seletiva_urbana' in df.columns:
        df['coleta_seletiva_rural'] = df['coleta_seletiva_total'] - df['coleta_seletiva_urbana']
        df['coleta_seletiva_rural'] = df['coleta_seletiva_rural'].clip(lower=0)
    
    # 5. Cobertura rural estimada
    if 'cobertura_total' in df.columns and 'cobertura_urbana' in df.columns:
        df['cobertura_rural_estimada'] = df['cobertura_total'] - df['cobertura_urbana']
        df['cobertura_rural_estimada'] = df['cobertura_rural_estimada'].clip(lower=0)
    
    # 6. Lacuna de coleta seletiva
    if 'cobertura_total' in df.columns and 'coleta_seletiva_total' in df.columns:
        df['lacuna_coleta_seletiva'] = df['cobertura_total'] - df['coleta_seletiva_total']
        df['lacuna_coleta_seletiva'] = df['lacuna_coleta_seletiva'].clip(lower=0)
    
    # 7. Qualidade do serviço (proporção de coleta seletiva)
    if 'coleta_seletiva_total' in df.columns and 'cobertura_total' in df.columns:
        df['qualidade_servico'] = (df['coleta_seletiva_total'] / (df['cobertura_total'] + 1)) * 100
        df['qualidade_servico'] = df['qualidade_servico'].clip(upper=100)
    
    # 8. Eficiência da coleta direta
    if 'cobertura_urbana_direta' in df.columns and 'cobertura_urbana' in df.columns:
        df['eficiencia_coleta_direta'] = df['cobertura_urbana_direta'] / (df['cobertura_urbana'] + 1) * 100
        df['eficiencia_coleta_direta'] = df['eficiencia_coleta_direta'].clip(upper=100)
    
    # 9. Cobertura média (indicador geral)
    cobertura_cols = ['cobertura_total', 'cobertura_urbana', 'cobertura_rural']
    cobertura_cols = [c for c in cobertura_cols if c in df.columns]
    if cobertura_cols:
        df['cobertura_media'] = df[cobertura_cols].mean(axis=1)
    
    # 10. Score de infraestrutura (combinação de indicadores)
    if 'cobertura_total' in df.columns and 'coleta_seletiva_total' in df.columns:
        # Normalizar cada componente (0-1)
        cobertura_norm = df['cobertura_total'] / 100
        coleta_seletiva_norm = df['coleta_seletiva_total'] / 100
        
        # Score ponderado (cobertura tem peso maior)
        df['infraestrutura_score'] = (cobertura_norm * 0.6) + (coleta_seletiva_norm * 0.4)
        df['infraestrutura_score'] = df['infraestrutura_score'] * 100
    
    return df


def create_sanitation_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features binárias a partir de limiares de saneamento.
    """
    df = df.copy()
    
    # 1. Alta cobertura (> 80%)
    if 'cobertura_total' in df.columns:
        df['cobertura_alta'] = (df['cobertura_total'] > 80).astype(int)
        df['cobertura_muito_alta'] = (df['cobertura_total'] > 95).astype(int)
        df['cobertura_baixa'] = (df['cobertura_total'] < 50).astype(int)
    
    # 2. Coleta seletiva presente (> 0%)
    if 'coleta_seletiva_total' in df.columns:
        df['tem_coleta_seletiva'] = (df['coleta_seletiva_total'] > 0).astype(int)
        df['coleta_seletiva_alta'] = (df['coleta_seletiva_total'] > 50).astype(int)
    
    # 3. Densidade alta (> 100 hab/km²)
    if 'densidade_populacional' in df.columns:
        df['densidade_alta'] = (df['densidade_populacional'] > 100).astype(int)
    
    # 4. Área urbana predominante (> 70% urbana)
    if 'proporcao_urbana' in df.columns:
        df['predominantemente_urbano'] = (df['proporcao_urbana'] > 0.7).astype(int)
    
    # 5. Qualidade do serviço boa (> 20% de coleta seletiva)
    if 'qualidade_servico' in df.columns:
        df['qualidade_servico_boa'] = (df['qualidade_servico'] > 20).astype(int)
    
    return df


def create_sanitation_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features temporais para dados de saneamento.
    """
    df = df.copy()
    
    # 1. Tendência de cobertura (variação ano a ano)
    if 'cobertura_total' in df.columns and 'year' in df.columns and 'ibge_code' in df.columns:
        df = df.sort_values(['ibge_code', 'year'])
        
        # Mudança percentual na cobertura
        df['cobertura_pct_change'] = df.groupby('ibge_code')['cobertura_total'].pct_change()
        df['cobertura_pct_change'] = df['cobertura_pct_change'].replace([np.inf, -np.inf], np.nan)
        
        # Diferença absoluta
        df['cobertura_diff'] = df.groupby('ibge_code')['cobertura_total'].diff()
        
        # Tendência de melhora (cobertura aumentou)
        df['cobertura_melhorou'] = (df['cobertura_diff'] > 0).astype(int)
    
    # 2. Tendência de coleta seletiva
    if 'coleta_seletiva_total' in df.columns and 'year' in df.columns and 'ibge_code' in df.columns:
        df['coleta_seletiva_pct_change'] = df.groupby('ibge_code')['coleta_seletiva_total'].pct_change()
        df['coleta_seletiva_pct_change'] = df['coleta_seletiva_pct_change'].replace([np.inf, -np.inf], np.nan)
        
        df['coleta_seletiva_diff'] = df.groupby('ibge_code')['coleta_seletiva_total'].diff()
        df['coleta_seletiva_melhorou'] = (df['coleta_seletiva_diff'] > 0).astype(int)
    
    # 3. Anos desde a última melhoria significativa
    if 'cobertura_diff' in df.columns and 'ibge_code' in df.columns:
        def years_since_improvement(series):
            result = []
            last_improvement = -1
            for i, val in enumerate(series):
                if val > 0:  # melhorou
                    last_improvement = i
                    result.append(0)
                else:
                    if last_improvement == -1:
                        result.append(999)
                    else:
                        result.append(i - last_improvement)
            return result
        
        df['anos_sem_melhoria'] = df.groupby('ibge_code')['cobertura_diff'].transform(
            lambda x: years_since_improvement(x.tolist())
        )
    
    return df


def create_sanitation_combined_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features combinadas que integram múltiplos indicadores de saneamento.
    """
    df = df.copy()
    
    # 1. Índice de Desenvolvimento do Saneamento (IDS)
    # Combina cobertura, coleta seletiva e qualidade
    if all(c in df.columns for c in ['cobertura_total', 'coleta_seletiva_total', 'qualidade_servico']):
        # Normalizar cada componente (0-1)
        cobertura_norm = df['cobertura_total'] / 100
        seletiva_norm = df['coleta_seletiva_total'] / 100
        qualidade_norm = df['qualidade_servico'] / 100
        
        # IDS com pesos
        df['ids'] = (cobertura_norm * 0.4 + seletiva_norm * 0.35 + qualidade_norm * 0.25) * 100
    
    # 2. Índice de Cobertura Efetiva
    # Cobertura ajustada pela proporção urbana
    if 'cobertura_total' in df.columns and 'proporcao_urbana' in df.columns:
        df['cobertura_efetiva'] = df['cobertura_total'] * (0.7 + 0.3 * df['proporcao_urbana'])
    
    # 3. Score de Risco de Saneamento (quanto maior, pior o saneamento)
    # Baseado em lacunas de cobertura e baixa coleta seletiva
    risk_components = []
    
    if 'lacuna_coleta_seletiva' in df.columns:
        risk_components.append(df['lacuna_coleta_seletiva'] / 100)
    
    if 'cobertura_total' in df.columns:
        risk_components.append(1 - df['cobertura_total'] / 100)
    
    if risk_components:
        df['risco_saneamento'] = np.mean(risk_components, axis=0) * 100
    
    return df


# ============================================================================
# SANITATION FEATURE ENGINEERING PIPELINE
# ============================================================================

def sanitation_feature_engineering_pipeline(
    df: pd.DataFrame,
    create_derived: bool = True,
    create_binary: bool = True,
    create_temporal: bool = True,
    create_combined: bool = True
) -> pd.DataFrame:
    """
    Pipeline completo de Feature Engineering para dados de saneamento.
    """
    print("1. Criando features derivadas de saneamento...")
    if create_derived:
        df = create_sanitation_derived_features(df)
    
    print("2. Criando features binárias de saneamento...")
    if create_binary:
        df = create_sanitation_binary_features(df)
    
    print("3. Criando features temporais de saneamento...")
    if create_temporal:
        df = create_sanitation_temporal_features(df)
    
    print("4. Criando features combinadas de saneamento...")
    if create_combined:
        df = create_sanitation_combined_features(df)
    
    print(f"✓ Feature Engineering de saneamento concluído. Shape final: {df.shape}")
    
    return df


# ============================================================================
# VALIDATION
# ============================================================================

def validate_sanitation_features(df: pd.DataFrame) -> dict:
    """
    Valida as features de saneamento criadas.
    """
    # Features esperadas
    derived_features = [
        'densidade_populacional', 'proporcao_urbana', 'proporcao_rural',
        'moradores_por_domicilio', 'coleta_seletiva_rural', 'cobertura_rural_estimada',
        'lacuna_coleta_seletiva', 'qualidade_servico', 'eficiencia_coleta_direta',
        'cobertura_media', 'infraestrutura_score'
    ]
    
    binary_features = [
        'cobertura_alta', 'cobertura_muito_alta', 'cobertura_baixa',
        'tem_coleta_seletiva', 'coleta_seletiva_alta',
        'densidade_alta', 'predominantemente_urbano', 'qualidade_servico_boa'
    ]
    
    temporal_features = [
        'cobertura_pct_change', 'cobertura_diff', 'cobertura_melhorou',
        'coleta_seletiva_pct_change', 'coleta_seletiva_diff', 'coleta_seletiva_melhorou',
        'anos_sem_melhoria'
    ]
    
    combined_features = [
        'ids', 'cobertura_efetiva', 'risco_saneamento'
    ]
    
    validation = {
        'shape': df.shape,
        'n_features': len(df.columns),
        'n_missing': df.isnull().sum().sum(),
        'features': {
            'derivadas': [c for c in derived_features if c in df.columns],
            'binarias': [c for c in binary_features if c in df.columns],
            'temporais': [c for c in temporal_features if c in df.columns],
            'combinadas': [c for c in combined_features if c in df.columns]
        }
    }
    
    return validation


def print_sanitation_feature_summary(validation: dict):
    """Imprime sumário das features de saneamento criadas."""
    print("\n" + "="*50)
    print("SANITATION FEATURE ENGINEERING SUMMARY")
    print("="*50)
    print(f"Shape:              {validation['shape']}")
    print(f"Total Features:     {validation['n_features']}")
    print(f"Missing Values:     {validation['n_missing']}")
    print("\n--- FEATURES BY CATEGORY ---")
    
    category_names = {
        'derivadas': 'Features Derivadas',
        'binarias': 'Features Binárias',
        'temporais': 'Features Temporais',
        'combinadas': 'Features Combinadas'
    }
    
    for key, name in category_names.items():
        features = validation['features'].get(key, [])
        if features:
            print(f"\n{name}: {len(features)} features")
            if len(features) <= 5:
                print(f"  {features}")
            else:
                print(f"  {features[:3]} ... +{len(features)-3} more")
    print("="*50)