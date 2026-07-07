"""Phase 4: Live Execution & Kotak Neo Integration.

Wraps `neo_api_client` for authentication (TOTP, no manual login), tick
streaming via WebSocket, and MIS order routing.

SAFETY: order placement is a no-op paper trade unless `settings.live_trading`
is True (env: LIVE_TRADING=true). This is intentional so the bot can be
wired end-to-end and observed before it is trusted with real capital.
"""
import logging
import time
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


@dataclass
class FillReport:
    """What the broker actually confirms for an order, vs. what we assumed
    when place_order() returned. See README "Known limitations" -- the system
    previously trusted its own in-memory state and last-seen tick price
    rather than ever polling order_history for the confirmed fill."""
    order_id: str
    order_status: str
    filled_quantity: int
    avg_price: float | None
    matches_expected: bool


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

    def get_order_status(self, order_id: str) -> dict:
        """Poll order_history for this order's current broker-side state
        (ordSt, fldQty, avgPrc, rlzPL). Field names per README "Known
        limitations" -- confirm against a live session before trusting them
        in a way that blocks trading, since they were never verified end to
        end against a real fill.

        order_history returns a list of status updates (oldest first); the
        last entry is the most current state.
        """
        response = self.client.order_history(order_id=order_id)
        data = response.get("data", response)
        if isinstance(data, list):
            return data[-1] if data else {}
        return data if isinstance(data, dict) else {}


def reconcile_fill(client: "KotakNeoClient", order_id: str, expected_quantity: int,
                    max_attempts: int = 3, poll_interval_seconds: float = 1.0) -> FillReport:
    """Confirm what the broker actually filled instead of trusting place_order's
    immediate response and our own last-seen tick price (see README "Known
    limitations": "we never poll the broker for confirmed fills").

    Paper orders always match by construction -- there's no broker fill to
    confirm against, so this is a no-op for the default (non-live) path.

    Retries a few times with a short pause: NSE order confirmation isn't
    always instantaneous, and polling once immediately after placement can
    race the broker's own processing.
    """
    if order_id.startswith("PAPER-"):
        return FillReport(order_id=order_id, order_status="simulated",
                           filled_quantity=expected_quantity, avg_price=None, matches_expected=True)

    status: dict = {}
    for attempt in range(max_attempts):
        status = client.get_order_status(order_id)
        if status.get("fldQty") not in (None, "", "0", 0):
            break
        if attempt < max_attempts - 1:
            time.sleep(poll_interval_seconds)

    filled_quantity = int(status.get("fldQty") or 0)
    avg_price = float(status["avgPrc"]) if status.get("avgPrc") not in (None, "") else None
    order_status = str(status.get("ordSt", "unknown"))
    matches_expected = filled_quantity == expected_quantity

    if not matches_expected:
        logger.warning(
            "Fill mismatch on order %s: expected qty %d, broker reports %d filled (status=%s)",
            order_id, expected_quantity, filled_quantity, order_status,
        )

    return FillReport(order_id=order_id, order_status=order_status, filled_quantity=filled_quantity,
                       avg_price=avg_price, matches_expected=matches_expected)


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
