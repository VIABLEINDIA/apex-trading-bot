# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## [v1.1] - 2026-07-07

[Release](https://github.com/VIABLEINDIA/apex-trading-bot/releases/tag/v1.1)

Test coverage, CI, dependency pinning, broker fill reconciliation, and ADRs
for the risk-management decisions, on top of the initial architecture
blueprint.

### Added
- GitHub Actions CI (`.github/workflows/tests.yml`) running pytest on push/PR
- Unit tests for all 12 `src` modules (64 -> 134 tests), including
  `execution.py` and `live_session.py` against a mocked broker
- Broker fill reconciliation (`execution.reconcile_fill`, wired into
  `live_session.py`): a BUY order that doesn't fill no longer silently opens
  a position, and a SELL's exit price uses the broker-confirmed `avgPrc`
  when available instead of trusting the last-seen tick price
- ADRs in `docs/decisions/` documenting the circuit-breaker-vs-stop-loss
  decision, sector/hourly/capital-shield limits, the BALU ATR-stop
  mechanism (flagged as an unproven regression), and JEANS' consecutive-loss
  halt/kill switch
- CI status badge in README
- Branch protection on `master` (required `pytest` status check, no
  force-push/deletion)

### Changed
- `requirements.txt`/`requirements-live.txt` pinned to exact versions
  (previously `>=` ranges) so the walk-forward benchmark numbers in README
  stay reproducible

## [v1.0] - 2026-07-07

Initial commit: full-stack NSE intraday momentum bot -- Nifty 500 screener,
GradientBoostingClassifier signal engine, portfolio risk manager (circuit
breaker, sector/hourly limits, capital shield, ATR stop, kill switch,
consecutive-loss halt), Kotak Neo execution layer, SQLite journal, and a
two-venv architecture. Includes walk-forward validation tooling, realistic
cost modeling, and parameter/model sweep scripts. Model's predictive edge
(~52% holdout accuracy) not yet validated as profitable net of realistic
costs -- see README "Model performance".
