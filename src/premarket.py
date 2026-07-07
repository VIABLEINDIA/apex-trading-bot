"""Pre-market step (08:45 IST): run the momentum screener and hand the
watchlist off to the live-execution process via a small JSON file.

Runs in the MAIN venv (the one with yfinance). This is deliberately a
separate, short-lived script from the live execution loop (src/live_session.py):
neo_api_client pins websockets==8.1 / certifi==2022.12.7, which conflicts with
yfinance's websockets>=13 requirement, so the two can never share one Python
environment. See requirements-live.txt and scripts/deploy/crontab.txt for how
cron wires the two processes together.
"""
import json
import logging
from pathlib import Path

from src.config import settings
from src.screener import run_screener

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

WATCHLIST_PATH = Path(settings.db_path).parent / "watchlist.json"


def main() -> None:
    logger.info("Running pre-market momentum screener...")
    selected = run_screener()
    watchlist = selected["ticker"].tolist() if not selected.empty else []

    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    WATCHLIST_PATH.write_text(json.dumps({"tickers": watchlist}))
    logger.info("Watchlist ready: %d tickers, written to %s", len(watchlist), WATCHLIST_PATH)


if __name__ == "__main__":
    main()
