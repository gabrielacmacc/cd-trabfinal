
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