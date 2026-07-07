"""Central configuration loaded from environment variables (.env)."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class KotakNeoConfig:
    consumer_key: str = os.getenv("KOTAK_CONSUMER_KEY", "")
    consumer_secret: str = os.getenv("KOTAK_CONSUMER_SECRET", "")
    mobile_number: str = os.getenv("KOTAK_MOBILE_NUMBER", "")
    password: str = os.getenv("KOTAK_PASSWORD", "")
    totp_secret: str = os.getenv("KOTAK_TOTP_SECRET", "")
    neo_fin_key: str = os.getenv("KOTAK_NEO_FIN_KEY", "neotrade")
    # Real Kotak Neo auth is a two-step TOTP flow (totp_login then
    # totp_validate), which needs the client's UCC (unique client code) and a
    # numeric MPIN separate from the account password -- see execution.py.
    ucc: str = os.getenv("KOTAK_UCC", "")
    mpin: str = os.getenv("KOTAK_MPIN", "")


@dataclass(frozen=True)
class RiskConfig:
    total_capital: float = float(os.getenv("TOTAL_CAPITAL", "100000"))
    max_slots: int = int(os.getenv("MAX_SLOTS", "10"))
    capital_per_slot_pct: float = float(os.getenv("CAPITAL_PER_SLOT_PCT", "0.10"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
    daily_loss_limit_pct: float = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.02"))
    max_positions_per_sector: int = int(os.getenv("MAX_POSITIONS_PER_SECTOR", "3"))
    max_trades_per_hour: int = int(os.getenv("MAX_TRADES_PER_HOUR", "4"))

    # ATR-based stop-loss + trailing profit-lock (see src/portfolio.py
    # PortfolioManager.check_exits). Ratios adapted from the BALU strategy
    # profile (backtest_atr_sl_mult / trail_activation_atr / trail_distance_atr).
    atr_stop_mult: float = float(os.getenv("ATR_STOP_MULT", "0.75"))
    atr_trail_activation_mult: float = float(os.getenv("ATR_TRAIL_ACTIVATION_MULT", "0.25"))
    atr_trail_distance_mult: float = float(os.getenv("ATR_TRAIL_DISTANCE_MULT", "0.12"))

    # Risk-based position sizing: risk this fraction of total capital per
    # trade (based on stop distance), capped by the flat slot-capital
    # allocation as a safety ceiling. See PortfolioManager.evaluate_signal.
    risk_per_trade_pct: float = float(os.getenv("RISK_PER_TRADE_PCT", "0.013"))

    # Trading-window session gating: adapted from BALU's 09:00-10:00 primary /
    # 10:00-13:15 continuation / no pre-close structure, shifted to NSE's
    # actual 09:15 open. Our own time-of-day slippage table already shows the
    # close is the worst time to trade -- this avoids it entirely rather than
    # just pricing in the extra cost.
    primary_session_end: str = os.getenv("PRIMARY_SESSION_END", "10:15")
    continuation_session_end: str = os.getenv("CONTINUATION_SESSION_END", "13:15")
    continuation_confidence_bonus: float = float(os.getenv("CONTINUATION_CONFIDENCE_BONUS", "0.05"))
    continuation_size_mult: float = float(os.getenv("CONTINUATION_SIZE_MULT", "0.4"))

    # Adapted from JEANS (third-generation rewrite in the same lineage as
    # APEXBOT/BALU): halt new entries after this many losing trades in a row
    # (resets daily, same as the other per-day counters), independent of
    # whether cumulative daily PnL has breached DAILY_LOSS_LIMIT_PCT yet.
    max_consecutive_losses: int = int(os.getenv("MAX_CONSECUTIVE_LOSSES", "5"))

    # Manual kill switch: if this file exists, no new entries are placed on
    # the next evaluate_signal call (existing positions still exit normally).
    # Checked fresh each call -- no polling/caching -- so removing the file
    # resumes trading immediately.
    kill_switch_path: str = os.getenv("KILL_SWITCH_PATH", "./data/KILL_SWITCH")


@dataclass(frozen=True)
class ScreenerConfig:
    universe_size: int = int(os.getenv("SCREENER_UNIVERSE_SIZE", "500"))
    top_n: int = int(os.getenv("SCREENER_TOP_N", "300"))
    roc_lookback_days: int = int(os.getenv("ROC_LOOKBACK_DAYS", "30"))


@dataclass(frozen=True)
class Settings:
    live_trading: bool = _bool("LIVE_TRADING", False)
    db_path: str = os.getenv("DB_PATH", "./data/trading_journal.db")
    model_path: str = os.getenv("MODEL_PATH", "./models/momentum_classifier.joblib")
    kotak: KotakNeoConfig = field(default_factory=KotakNeoConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)


settings = Settings()
