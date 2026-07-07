"""Connectivity-only smoke test for Kotak Neo: login, TOTP 2FA, and a short
WebSocket tick subscription. Places ZERO orders -- it never calls
KotakNeoClient.place_order, so it's safe to run regardless of the
LIVE_TRADING setting in .env.

Run this in venv-live (needs neo_api_client) after filling in real
credentials in .env, to confirm auth + streaming actually work before trusting
src/live_session.py with them:

    venv-live/bin/python scripts/test_broker_connection.py
    venv-live/bin/python scripts/test_broker_connection.py --seconds 60 RELIANCE.NS TCS.NS

Only run this during NSE market hours (09:15-15:30 IST on a trading day) --
outside that window the WebSocket may connect but no ticks will arrive, which
would look like a failure but isn't one.
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.execution import KotakNeoClient  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TICKERS = ["RELIANCE.NS", "TCS.NS"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS, help="Tickers to subscribe to for the test.")
    parser.add_argument("--seconds", type=int, default=30, help="How long to listen for ticks before exiting.")
    args = parser.parse_args()

    cfg = settings.kotak
    # Note: KOTAK_PASSWORD isn't required here -- the verified totp_login/
    # totp_validate flow authenticates with UCC + TOTP + MPIN, not the account
    # password (see src/execution.py KotakNeoClient.login). KOTAK_CONSUMER_SECRET
    # also isn't required -- the installed neo_api_client's NeoAPI.__init__ has
    # no such parameter at all (see KotakNeoClient.__init__'s comment).
    missing = [name for name, val in [
        ("KOTAK_CONSUMER_KEY", cfg.consumer_key),
        ("KOTAK_MOBILE_NUMBER", cfg.mobile_number), ("KOTAK_TOTP_SECRET", cfg.totp_secret),
        ("KOTAK_UCC", cfg.ucc), ("KOTAK_MPIN", cfg.mpin),
    ] if not val]
    if missing:
        logger.error("Missing required .env fields: %s. Fill these in before running this test.", ", ".join(missing))
        return

    tick_count = 0

    def on_tick(message: dict) -> None:
        nonlocal tick_count
        tick_count += 1
        if tick_count <= 5 or tick_count % 20 == 0:
            logger.info("Tick #%d: %s", tick_count, message)

    logger.info("Connecting to Kotak Neo (this places NO orders, regardless of LIVE_TRADING=%s)...", settings.live_trading)
    client = KotakNeoClient()

    try:
        client.login()
        logger.info("Login + 2FA succeeded.")
    except Exception:
        logger.exception("Login failed -- check credentials/TOTP secret in .env and Kotak Neo developer console status.")
        return

    try:
        client.subscribe_ticks(args.tickers, on_tick=on_tick)
        logger.info("Subscribed to %s. Listening for %ds...", args.tickers, args.seconds)
    except Exception:
        logger.exception("Subscribe failed -- login worked but the WebSocket subscription did not.")
        return

    time.sleep(args.seconds)

    if tick_count == 0:
        logger.warning(
            "No ticks received in %ds. If it's outside NSE market hours (09:15-15:30 IST, Mon-Fri) "
            "this is expected. Otherwise, check the subscribe_ticks() message format in src/execution.py "
            "against the current Kotak Neo docs -- the field names there are unverified.",
            args.seconds,
        )
    else:
        logger.info("SUCCESS: received %d ticks. Login, 2FA, and streaming all work.", tick_count)


if __name__ == "__main__":
    main()
