# Project Overview
- Name: 深度量化0927 — a professional quantitative trading research workspace.
- Version: v1.1 (架构更新 2025-11-30) | v1.0 策略封板 (2025-11-28)
- Purpose: ETF rotation strategy research platform with three-tier engine architecture (WFO → VEC → BT).
- **Priority subsystem**: `src/etf_strategy/` (production-ready ETF rotation with 43 ETFs, 18 factors, 12,597 combinations).
- **Best strategy** (locked): CORRELATION_TO_MARKET_20D + MAX_DD_60D + PRICE_POSITION_120D + PRICE_POSITION_20D = **121.02% returns** (verified reproducible).
- Core principles: pragmatic engineering, no special casing, production reliability, no-leakage data handling, VEC/BT alignment (<0.01pp), comprehensive documentation.
- **Important**: All LLM agents MUST read `AGENTS.md` before working on this project!
- Maintainer: 深度量化团队 (quant engineering team).
