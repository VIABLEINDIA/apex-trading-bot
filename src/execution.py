"""Phase 4: Live Execution & Kotak Neo Integration.

Wraps `neo_api_client` for authentication (TOTP, no manual login), tick
streaming via WebSocket, and MIS order routing.

SAFETY: order placement is a no-op paper trade unless `settings.live_trading`
is True (env: LIVE_TRADING=true). This is intentional so the bot can be
wired end-to-end and observed before it is trusted with real capital.
"""
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import pyotp

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class OrderResult:
    order_id: str
    status: str
    paper: bool


def to_kotak_trading_symbol(yahoo_ticker: str) -> str:
    """Convert our internal yfinance-style ticker ("RELIANCE.NS") to Kotak
    Neo's trading_symbol format ("RELIANCE-EQ"). Every other module
    (screener, model, portfolio, database) uses the yfinance format
    consistently; only the Kotak Neo API boundary needs this conversion."""
    return f"{yahoo_ticker.removesuffix('.NS')}-EQ"


class KotakNeoClient:
    """Thin wrapper around neo_api_client.NeoAPI.

    Imported lazily inside __init__ so the rest of the codebase (and tests)
    can run without the neo_api_client package / real credentials installed.
    """

    def __init__(self):
        from neo_api_client import NeoAPI  # lazy import, requires real creds to construct

        cfg = settings.kotak
        # NeoAPI.__init__ genuinely has no consumer_secret parameter in the
        # installed neo_api_client (confirmed by reading the installed
        # package: the signature is (environment, access_token, neo_fin_key,
        # consumer_key) only, and the body has consumer_secret commented out
        # entirely) -- passing it raises TypeError for an unexpected keyword
        # argument. KOTAK_CONSUMER_SECRET is kept in config for now in case a
        # different neo_api_client version needs it, but isn't passed here.
        self.client = NeoAPI(
            consumer_key=cfg.consumer_key,
            environment="prod",
            neo_fin_key=cfg.neo_fin_key,
        )
        self._logged_in = False

    def login(self) -> None:
        """Two-step TOTP auth: totp_login (mobile + UCC + TOTP code) gets a
        session id, then totp_validate (MPIN) exchanges it for an auth token.
        Both responses may arrive wrapped as {"data": {...}} or flat -- unwrap
        defensively either way.
        """
        cfg = settings.kotak
        totp_code = pyotp.TOTP(cfg.totp_secret).now()

        login_response = self.client.totp_login(
            mobile_number=cfg.mobile_number, ucc=cfg.ucc, totp=totp_code,
        )
        login_data = login_response.get("data", login_response)
        _session_id = login_data["sid"]  # noqa: F841 -- consumed internally by the SDK client

        validate_response = self.client.totp_validate(mpin=cfg.mpin)
        validate_data = validate_response.get("data", validate_response)
        _auth_token = validate_data["token"]  # noqa: F841 -- consumed internally by the SDK client

        self._logged_in = True
        logger.info("Kotak Neo login + TOTP validation complete.")

    def subscribe_ticks(self, instrument_tokens: list[str], on_tick: Callable[[dict], None],
                        is_index: bool = False) -> None:
        """Subscribe to the WebSocket feed; `on_tick` is called for every message.

        Can be called more than once (e.g. once for the stock watchlist, once
        with `is_index=True` for a benchmark index) -- each call re-sets
        `on_message` to an equivalent closure calling the same `on_tick`, and
        neo_api_client's `subscribe()` is expected to add to existing
        subscriptions rather than replace them, though this hasn't been
        verified against a live session (see README "Known limitations").
        """
        if not self._logged_in:
            raise RuntimeError("login() must be called before subscribing to ticks")

        def _on_message(message):
            on_tick(message)

        self.client.on_message = _on_message
        self.client.subscribe(instrument_tokens=instrument_tokens, isIndex=is_index, isDepth=False)
        logger.info("Subscribed to %d instrument tokens (isIndex=%s).", len(instrument_tokens), is_index)

    def place_order(self, ticker: str, quantity: int, transaction_type: str = "BUY",
                     order_type: str = "MKT", product: str = "MIS") -> OrderResult:
        """`ticker` is our internal yfinance-style symbol ("RELIANCE.NS");
        `transaction_type` is "BUY"/"SELL" (our convention throughout the
        codebase) -- both get translated to Kotak Neo's actual wire format
        (trading_symbol "RELIANCE-EQ", transaction_type "B"/"S") right here,
        so callers don't need to know about it.
        """
        if not settings.live_trading:
            paper_id = f"PAPER-{uuid.uuid4().hex[:10]}"
            logger.info(
                "[PAPER TRADE] %s %s x%d (product=%s, order_type=%s) -> %s",
                transaction_type, ticker, quantity, product, order_type, paper_id,
            )
            return OrderResult(order_id=paper_id, status="simulated", paper=True)

        kotak_transaction_type = "B" if transaction_type == "BUY" else "S"
        response = self.client.place_order(
            exchange_segment="nse_cm",
            product=product,
            price="0",
            order_type=order_type,
            quantity=str(quantity),
            validity="DAY",
            transaction_type=kotak_transaction_type,
            trading_symbol=to_kotak_trading_symbol(ticker),
            amo="NO",
        )
        order_id = response.get("nOrdNo", "UNKNOWN")
        logger.warning("[LIVE ORDER] %s %s x%d -> order_id=%s", transaction_type, ticker, quantity, order_id)
        return OrderResult(order_id=order_id, status=response.get("stat", "unknown"), paper=False)


class TickBarAggregator:
    """Resamples raw tick messages into rolling 15-minute OHLCV bars per ticker,
    so the AI engine always has a fresh DataFrame to featurize.
    """

    def __init__(self, bar_interval: str = "15min"):
        self.bar_interval = bar_interval
        self._ticks: dict[str, list[dict]] = defaultdict(list)

    def add_tick(self, ticker: str, price: float, volume: float, timestamp: pd.Timestamp) -> None:
        self._ticks[ticker].append({"timestamp": timestamp, "price": price, "volume": volume})

    def get_bars(self, ticker: str) -> pd.DataFrame:
        ticks = self._ticks.get(ticker, [])
        if not ticks:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df = pd.DataFrame(ticks).set_index("timestamp")
        bars = df["price"].resample(self.bar_interval).ohlc()
        bars["Volume"] = df["volume"].resample(self.bar_interval).sum()
        bars.columns = ["Open", "High", "Low", "Close", "Volume"]
        return bars.dropna(subset=["Close"])
