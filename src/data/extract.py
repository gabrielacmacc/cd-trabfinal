import shutil
import time
from io import StringIO
from pathlib import Path

import kagglehub
import pandas as pd
import requests

DATA_RAW = Path(__file__).parents[2] / "data" / "raw"
DATA_PROCESSED = Path(__file__).parents[2] / "data" / "processed"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# --- INMET (meteorological data) ---

def fetch_inmet(years: range = range(2020, 2025)) -> pd.DataFrame:
    dest = DATA_RAW / "inmet" / "inmet_rs_2020_2024.csv"
    raw_dir = DATA_RAW / "inmet"

    if dest.exists():
        print(f"INMET already at: {dest}")
        return pd.read_csv(dest, low_memory=False)

    csv_files = sorted(raw_dir.glob("*.csv")) if raw_dir.exists() else []
    if not csv_files:
        src = Path(kagglehub.dataset_download("gnomows/dados-metereologicos-2018-2024-inmet"))
        shutil.copytree(src, raw_dir)
        csv_files = sorted(raw_dir.glob("*.csv"))

    frames = []
    for path in csv_files:
        try:
            year = int(path.stem)
        except ValueError:
            continue
        if year not in years:
            continue
        df = pd.read_csv(path, low_memory=False)
        df = df[df["UF"] == "RS"].copy()
        df["year"] = year
        frames.append(df)
        print(f"  Loaded INMET {year}: {len(df):,} rows")

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(dest, index=False)
    print(f"INMET saved to: {dest}  ({len(result):,} rows, {result['ESTACAO'].nunique()} stations)")
    return result

# --- IBGE (city codes) ---

_IBGE_RS_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/43/municipios"

def fetch_rs_geocodes() -> list[dict]:
    dest = DATA_RAW / "ibge" / "rs_city_codes.csv"
    if dest.exists():
        print(f"IBGE city codes already at: {dest}")
        return pd.read_csv(dest).to_dict("records")
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(_IBGE_RS_URL, timeout=10)
    r.raise_for_status()
    cities = r.json()
    pd.DataFrame(cities)[["id", "nome"]].to_csv(dest, index=False)
    print(f"IBGE RS city codes saved to: {dest}  ({len(cities)} municipalities)")
    return cities

# --- InfoDengue (dengue cases) ---

_INFODENGUE_URL = "https://info.dengue.mat.br/api/alertcity/"

def _fetch_dengue_city_year(geocode: int, year: int) -> pd.DataFrame:
    params = {
        "geocode": geocode,
        "disease": "dengue",
        "format": "csv",
        "ew_start": 1,
        "ew_end": 53,
        "ey_start": year,
        "ey_end": year,
    }
    r = requests.get(_INFODENGUE_URL, params=params, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def fetch_dengue(years: range = range(2020, 2025), delay: float = 0.3) -> pd.DataFrame:
    dest = DATA_RAW / "dengue" / "dengue_rs_2020_2024.csv"
    if dest.exists():
        print(f"InfoDengue already at: {dest}")
        return pd.read_csv(dest, low_memory=False)

    municipalities = fetch_rs_geocodes()
    frames = []

    for muni in municipalities:
        geocode = muni["id"]
        name = muni["nome"]
        for year in years:
            try:
                df = _fetch_dengue_city_year(geocode, year)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                print(f"  SKIP {name} ({geocode}) {year}: {e}")
            time.sleep(delay)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(dest, index=False)
    print(f"InfoDengue saved to: {dest}  ({len(result):,} rows, {result['municipio_nome'].nunique()} cities)")
    return result

# --- SNIS (sanitation data 2020-2022) ---

_SNIS_FILES = {
    2020: DATA_RAW / "residuos" / "Informacoes_RS_2020.csv",
    2021: DATA_RAW / "residuos" / "Informacoes_RS_2021.csv",
    2022: DATA_RAW / "residuos" / "Informacoes_RS_2022.csv",
}

def fetch_snis(year) -> pd.DataFrame:
    dest = DATA_RAW / "residuos" / f"Informacoes_RS_{year}.csv"
    if dest.exists():
        print(f"SNIS {year} already at: {dest}")
        result = pd.read_csv(dest, low_memory=False)
        result["year"] = year
        return result    

# --- SINISA (solid waste data 2023-2024) ---

_SINISA_FILES = {
    2023: DATA_RAW / "residuos" / "SINISA_RESIDUOS_Informacoes_Formulario_Limpeza_Urbana_2023_RS.csv",
    2024: DATA_RAW / "residuos" / "SINISA_RESIDUOS_Informacoes_Formulario_Limpeza_Urbana_2024_RS.csv",
}

def fetch_sinisa(years: range = range(2023, 2025)) -> pd.DataFrame:
    dest = DATA_RAW / "residuos" / "sinisa_rs_2023_2024.csv"
    if dest.exists():
        print(f"SINISA already at: {dest}")
        return pd.read_csv(dest, low_memory=False)

    frames = []
    for year in years:
        df = pd.read_csv(_SINISA_FILES[year], low_memory=False)
        # rename the year-specific survey column to a stable name
        df = df.rename(columns={c: "RESPONDEU_MODULO" for c in df.columns if c.startswith("RESPONDEU")})
        df["year"] = year
        frames.append(df)
        print(f"  Loaded SINISA {year}: {len(df)} municipalities, {df.shape[1] - 1} columns")

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(dest, index=False)
    print(f"SINISA saved to: {dest}  ({len(result):,} rows)")
    return result
