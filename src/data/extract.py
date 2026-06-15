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

def fetch_inmet() -> Path:
    dest = DATA_RAW / "inmet"
    if dest.exists():
        print(f"INMET already at: {dest}")
        return dest
    src = Path(kagglehub.dataset_download("gnomows/dados-metereologicos-2018-2024-inmet"))
    shutil.copytree(src, dest)
    print(f"INMET saved to: {dest}  ({len(list(dest.glob('*.csv')))} files)")
    return dest


# --- InfoDengue (dengue cases) ---

_INFODENGUE_URL = "https://info.dengue.mat.br/api/alertcity/"
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


def fetch_dengue(years: range = range(2020, 2026), delay: float = 0.3) -> pd.DataFrame:
    dest = DATA_RAW / "dengue" / "dengue_rs_2020_2025.csv"
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


# --- SNIS (sanitation data) ---
def fetch_snis():
    return