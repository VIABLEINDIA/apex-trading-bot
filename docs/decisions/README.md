# Architecture Decision Records

Records of significant, hard-to-reverse design decisions in this project and
why they were made. See individual ADRs for full context, alternatives
considered, and consequences.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-portfolio-circuit-breaker-not-per-trade-stop-loss.md) | Portfolio-level circuit breaker instead of a per-trade stop-loss | Accepted |
| [0002](0002-sector-hourly-capital-shield-limits.md) | Sector concentration, hourly trade budget, and graduated capital shield | Accepted |
| [0003](0003-atr-stop-trailing-session-gating-risk-sizing.md) | ATR stop / trailing profit-lock, session gating, and risk-based sizing (BALU) | Accepted (implementation) -- flagged as unproven regression |
| [0004](0004-consecutive-loss-halt-and-kill-switch.md) | Consecutive-loss halt and manual kill switch (JEANS) | Accepted |
| [0005](0005-cross-sectional-ranking-training-objective.md) | Cross-sectional ranking as an alternative training objective | Accepted (implementation) -- single-window result, unconfirmed |
