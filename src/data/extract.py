import re
import zipfile
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
INMET_ZIP_URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"

def _download_inmet_zip(year: int, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{year}.zip"
    if zip_path.exists():
        return zip_path
    url = INMET_ZIP_URL.format(year=year)
    print(f"  Baixando {url}")
    # alguns ambientes têm problema de SSL com esse host; se necessário,
    # tente verify=False como último recurso (não recomendado em produção)
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=180)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)
    return zip_path


def _extract_inmet_zip(zip_path: Path, year: int, raw_dir: Path) -> Path:
    extract_dir = raw_dir / str(year)
    if not extract_dir.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    return extract_dir


def _read_inmet_station_csv(path: Path) -> pd.DataFrame | None:
    """Lê um CSV de estação do INMET: 8 linhas de metadados + tabela."""
    with open(path, encoding="latin-1") as f:
        header_lines = [next(f) for _ in range(8)]

    meta = {}
    for line in header_lines:
        if ":;" in line:
            key, val = line.split(":;", 1)
            key = key.strip().upper()
            key = key.replace("ESTAÇÃO", "ESTACAO")  # normaliza acento
            meta[key] = val.strip().strip(";").strip()

    if meta.get("UF") != "RS":
        return None

    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="latin-1",
        skiprows=8,
        na_values=["-9999", "-9999,0", ""],
        low_memory=False,
    )
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]  # remove ; sobrando no final da linha

    df["UF"] = meta.get("UF")
    df["ESTACAO"] = meta.get("ESTACAO")
    df["CODIGO_WMO"] = meta.get("CODIGO (WMO)")
    df["LATITUDE"] = meta.get("LATITUDE")
    df["LONGITUDE"] = meta.get("LONGITUDE")
    df["ALTITUDE"] = meta.get("ALTITUDE")
    return df


def fetch_inmet(years: range = range(2020, 2025)) -> pd.DataFrame:
    dest = DATA_RAW / "inmet" / "inmet_rs_2020_2024.csv"
    raw_dir = DATA_RAW / "inmet"

    if dest.exists():
        print(f"INMET already at: {dest}")
        return pd.read_csv(dest, low_memory=False)

    frames = []
    for year in years:
        zip_path = _download_inmet_zip(year, raw_dir)
        extract_dir = _extract_inmet_zip(zip_path, year, raw_dir)

        # busca recursiva: alguns anos vêm com subpastas por estação/região
        csv_files = sorted(set(extract_dir.rglob("*.csv")) | set(extract_dir.rglob("*.CSV")))
        csv_files = [p for p in csv_files if "_RS_" in p.name.upper()]

        year_rows = 0
        for path in csv_files:
            df = _read_inmet_station_csv(path)
            if df is None:
                continue
            df["year"] = year
            frames.append(df)
            year_rows += len(df)
        print(f"  Loaded INMET {year}: {year_rows:,} rows ({len(csv_files)} estações RS)")

    result = pd.concat(frames, ignore_index=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
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


def fetch_cleaned_snis_sinisa() -> pd.DataFrame:
    dest = DATA_RAW / "residuos" / "cleaned_snis_sinisa_residuos_2020_2024.csv"
    
    if dest.exists():
        print(f"Cleaned SNIS/SINISA already at: {dest}")
        return pd.read_csv(dest, low_memory=False)
    
    # Se o arquivo não existir, tentar carregar do local alternativo
    # Ou você pode implementar a lógica de construção aqui
    st.error(f"Arquivo não encontrado: {dest}")
    st.info("""
    Certifique-se de que o arquivo `cleaned_snis_sinisa_residuos_2020_2024.csv` 
    está na pasta `data/raw/residuos/`
    """)
    return pd.DataFrame()
