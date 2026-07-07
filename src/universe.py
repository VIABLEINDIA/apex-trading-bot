"""NSE ticker universe + sector lookup, read from data/nifty500.csv.

Split out from screener.py because this has no yfinance dependency (just
pandas + a local CSV), unlike the rest of screener.py -- so it's safe to
import from src/live_session.py, which runs in the yfinance-free live venv
(see requirements-live.txt / README "Two venvs").
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
UNIVERSE_FILE = DATA_DIR / "nifty500.csv"

_SAMPLE_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT",
    "MARUTI", "TITAN", "SUNPHARMA", "ULTRACEMCO", "WIPRO", "NESTLEIND",
]


def load_universe() -> list[str]:
    """Load the NSE ticker universe (yfinance ".NS" suffixed) from data/nifty500.csv.

    The CSV must have a "Symbol" column, matching NSE's published Nifty 500 list
    (https://www.niftyindices.com -> Nifty 500 constituents). Falls back to a
    small bundled sample if the file is missing, so the pipeline is runnable
    out of the box for testing.
    """
    if UNIVERSE_FILE.exists():
        df = pd.read_csv(UNIVERSE_FILE)
        symbols = df["Symbol"].astype(str).str.strip().tolist()
    else:
        logger.warning(
            "data/nifty500.csv not found, falling back to bundled sample universe. "
            "Download the full Nifty 500 constituent list to data/nifty500.csv for production use."
        )
        symbols = _SAMPLE_UNIVERSE
    return [f"{s}.NS" for s in symbols]


def load_sector_map() -> dict[str, str]:
    """Load ticker -> Industry from data/nifty500.csv, for the portfolio
    manager's sector concentration limit. Returns {} if the file is missing
    (matching load_universe()'s fallback -- the sector limit just becomes a
    no-op if we don't know any sectors)."""
    if not UNIVERSE_FILE.exists():
        return {}
    df = pd.read_csv(UNIVERSE_FILE)
    symbols = df["Symbol"].astype(str).str.strip()
    industries = df["Industry"].astype(str).str.strip()
    return {f"{s}.NS": i for s, i in zip(symbols, industries)}
