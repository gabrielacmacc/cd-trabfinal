import unicodedata
import pandas as pd

# ---------------------------------------------------------------------------
# Column names
# ---------------------------------------------------------------------------

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip accents, replace spaces/special chars with underscores."""
    def _clean(name: str) -> str:
        name = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))
        name = name.lower().strip()
        name = "".join(c if c.isalnum() else "_" for c in name)
        while "__" in name:
            name = name.replace("__", "_")
        return name.strip("_")

    df = df.copy()
    df.columns = [_clean(c) for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# String / encoding
# ---------------------------------------------------------------------------

def fix_mojibake(series: pd.Series) -> pd.Series:
    """Fix strings that were UTF-8 encoded but decoded as Latin-1 (e.g. 'AceguÃ¡' -> 'Aceguá')."""
    def _fix(val):
        if not isinstance(val, str):
            return val
        try:
            return val.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return val

    return series.map(_fix)


def strip_accents(series: pd.Series) -> pd.Series:
    """Remove accents from a string series (e.g. 'Aceguá' -> 'Acegua')."""
    def _strip(val):
        if not isinstance(val, str):
            return val
        return "".join(
            c for c in unicodedata.normalize("NFKD", val)
            if not unicodedata.combining(c)
        )

    return series.map(_strip)


def normalize_city_name(series: pd.Series) -> pd.Series:
    """Standardize city names for joining across sources.

    Pipeline: fix mojibake -> strip accents -> uppercase -> collapse whitespace.
    Result is a join-key column, not a display column.
    """
    s = fix_mojibake(series)
    s = strip_accents(s)
    s = s.str.upper().str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


# ---------------------------------------------------------------------------
# Numeric parsing
# ---------------------------------------------------------------------------

def parse_float(series: pd.Series) -> pd.Series:
    """Convert numeric strings to float.

    Handles:
      '993,3'    -> 993.3   (decimal comma, no thousands sep)
      '1.500,00' -> 1500.0  (dot as thousands sep, comma as decimal)
      '4.170'    -> 4170.0  (dot as thousands sep, no decimal part)
      '0'        -> 0.0
    """
    s = series.astype(str).str.strip()
    has_both = s.str.contains(r"\.", regex=False) & s.str.contains(",", regex=False)
    only_comma = ~s.str.contains(r"\.", regex=False) & s.str.contains(",", regex=False)
    only_dot = s.str.contains(r"\.", regex=False) & ~s.str.contains(",", regex=False)

    result = s.copy()
    result[has_both] = s[has_both].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    result[only_comma] = s[only_comma].str.replace(",", ".", regex=False)
    result[only_dot] = s[only_dot].str.replace(".", "", regex=False)

    return pd.to_numeric(result, errors="coerce")


def cast_br_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect string columns that contain BR-formatted numbers and cast them to float.

    A column qualifies if: dtype is object/str, and at least 50% of non-null
    values match a BR numeric pattern (digits with optional . or , separators).
    """
    df = df.copy()
    br_pattern = r"^-?\d{1,3}(?:\.\d{3})*(?:,\d+)?$|^-?\d+(?:,\d+)?$"
    for col in df.select_dtypes(include=["object", "str"]).columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        match_rate = sample.astype(str).str.match(br_pattern).mean()
        if match_rate >= 0.5:
            df[col] = parse_float(df[col])
    return df


# ---------------------------------------------------------------------------
# Boolean parsing
# ---------------------------------------------------------------------------

def cast_sim_nao(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'Sim'/'Não' string columns to bool.

    Any column where values are 'Sim' or 'Não' (case-insensitive) is converted: 
    'sim' -> True, 'não'/'nao' -> False, everything else -> NaN (as bool NA).
    """
    MAP = {"sim": True, "não": False, "nao": False}

    df = df.copy()
    for col in df.select_dtypes(include=["object", "str"]).columns:
        sample = df[col].dropna().astype(str).str.strip().str.lower()
        if sample.empty:
            continue
        if (sample.isin(MAP).sum() / len(sample)) >= 0.8:
            df[col] = (
                df[col].astype(str).str.strip().str.lower()
                .map(MAP)
                .astype("boolean")
            )
    return df


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_dates(series: pd.Series, fmt: str | None = None) -> pd.Series:
    """Parse a date column.

    Common formats seen in this project:
      'YYYY/MM/DD'  (INMET)
      'YYYY-MM-DD'  (InfoDengue)
      'DD/MM/YY'    (INMET DATA DE FUNDACAO)
    """
    return pd.to_datetime(series, format=fmt, errors="coerce", dayfirst=False)


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values. Fill in strategy per column as needed."""
    return df


# ---------------------------------------------------------------------------
# Column dropping
# ---------------------------------------------------------------------------

def drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop columns by name, ignoring any that don't exist in the DataFrame."""
    return df.drop(columns=[c for c in columns if c in df.columns])


# ---------------------------------------------------------------------------
# Type splitting
# ---------------------------------------------------------------------------

def split_categorical_numerical(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into categorical and numerical subsets.

    Returns (categorical_df, numerical_df).
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    return df[cat_cols].copy(), df[num_cols].copy()
