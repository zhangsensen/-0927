"""
参数冻结模块 (frozen_params)

硬编码 v3.4 生产参数，防止配置漂移。
在 WFO / VEC / BT 三个执行入口加载配置后立即校验，不一致则 fast-fail。

用法:
    from etf_strategy.core.frozen_params import load_frozen_config

    config = yaml.safe_load(open("configs/combo_wfo_config.yaml"))
    frozen = load_frozen_config(config, config_path="configs/combo_wfo_config.yaml")
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strictness mode
# ---------------------------------------------------------------------------


class StrictnessMode(Enum):
    STRICT = "strict"
    WARN = "warn"


class FrozenParamViolation(Exception):
    """参数不一致时抛出"""

    def __init__(self, violations: List[str]):
        self.violations = violations
        msg = f"发现 {len(violations)} 个参数违规:\n" + "\n".join(
            f"  - {v}" for v in violations
        )
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Frozen dataclasses (全部 frozen=True)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenBacktestParams:
    freq: int = 3
    pos_size: int = 2
    commission_rate: float = 0.0002
    initial_capital: int = 1_000_000
    lookback_window: int = 252


@dataclass(frozen=True)
class FrozenTimingParams:
    enabled: bool = True
    type: str = "light_timing"
    extreme_threshold: float = -0.1
    extreme_position: float = 0.1
    index_timing_enabled: bool = False
    individual_timing_enabled: bool = False


@dataclass(frozen=True)
class FrozenRegimeGateParams:
    enabled: bool = True
    mode: str = "volatility"
    proxy_symbol: str = "510300"
    window: int = 20
    shift_days: int = 1
    thresholds_pct: Tuple[int, ...] = (25, 30, 40)
    exposures: Tuple[float, ...] = (1.0, 0.7, 0.4, 0.1)


@dataclass(frozen=True)
class FrozenRiskControlParams:
    enabled: bool = True
    leverage_cap: float = 1.0
    stop_check_on_rebalance_only: bool = True
    stop_method: str = "fixed"
    trailing_stop_pct: float = 0.0
    atr_window: int = 14
    atr_multiplier: float = 8.0
    cooldown_days: int = 0
    etf_stop_loss: float = 0.05


@dataclass(frozen=True)
class FrozenWFOParams:
    combo_sizes: Tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    enable_fdr: bool = True
    fdr_alpha: float = 0.05
    fdr_method: str = "fdr_bh"
    is_period: int = 180
    oos_period: int = 60
    step_size: int = 60
    top_n: int = 100_000
    rebalance_frequencies: Tuple[int, ...] = (3,)


@dataclass(frozen=True)
class FrozenScoringParams:
    complexity_penalty_lambda: float = 0.15
    use_robust_scoring: bool = True
    weights: Tuple[Tuple[str, float], ...] = (
        ("ann_ret", 0.4),
        ("max_dd", 0.3),
        ("sharpe", 0.3),
    )


@dataclass(frozen=True)
class FrozenCrossSectionParams:
    bounded_factors: Tuple[str, ...] = (
        "PRICE_POSITION_20D",
        "PRICE_POSITION_120D",
        "PV_CORR_20D",
        "RSI_14",
    )
    winsorize_lower: float = 0.025
    winsorize_upper: float = 0.975


@dataclass(frozen=True)
class FrozenETFPool:
    symbols: Tuple[str, ...] = (
        "159801",
        "159819",
        "159859",
        "159883",
        "159915",
        "159920",
        "159928",
        "159949",
        "159992",
        "159995",
        "159998",
        "510050",
        "510300",
        "510500",
        "511010",
        "511260",
        "511380",
        "512010",
        "512100",
        "512400",
        "512480",
        "512660",
        "512690",
        "512720",
        "512800",
        "512880",
        "512980",
        "513050",
        "513100",
        "513130",
        "513500",
        "515030",
        "515180",
        "515210",
        "515650",
        "515790",
        "516090",
        "516160",
        "516520",
        "518850",
        "518880",
        "588000",
        "588200",
    )
    qdii_codes: Tuple[str, ...] = (
        "159920",
        "513050",
        "513100",
        "513130",
        "513500",
    )

    @property
    def total_count(self) -> int:
        return len(self.symbols)

    @property
    def qdii_count(self) -> int:
        return len(self.qdii_codes)


@dataclass(frozen=True)
class FrozenStrategy:
    name: str
    factors: Tuple[str, ...]


@dataclass(frozen=True)
class FrozenProductionConfig:
    version: str
    config_sha256: Optional[str]
    backtest: FrozenBacktestParams
    timing: FrozenTimingParams
    regime_gate: FrozenRegimeGateParams
    risk_control: FrozenRiskControlParams
    wfo: FrozenWFOParams
    scoring: FrozenScoringParams
    cross_section: FrozenCrossSectionParams
    etf_pool: FrozenETFPool
    strategies: Tuple[FrozenStrategy, ...]


# ---------------------------------------------------------------------------
# v3.4 production config (硬编码)
# ---------------------------------------------------------------------------

_V3_4_CONFIG = FrozenProductionConfig(
    version="v3.4",
    config_sha256=None,  # 运行时计算
    backtest=FrozenBacktestParams(),
    timing=FrozenTimingParams(),
    regime_gate=FrozenRegimeGateParams(),
    risk_control=FrozenRiskControlParams(),
    wfo=FrozenWFOParams(),
    scoring=FrozenScoringParams(),
    cross_section=FrozenCrossSectionParams(),
    etf_pool=FrozenETFPool(),
    strategies=(
        FrozenStrategy(
            name="strategy_1",
            factors=("ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"),
        ),
        FrozenStrategy(
            name="strategy_2",
            factors=(
                "ADX_14D",
                "OBV_SLOPE_10D",
                "PRICE_POSITION_120D",
                "SHARPE_RATIO_20D",
                "SLOPE_20D",
            ),
        ),
    ),
)

_V4_0_CROSS_SECTION = FrozenCrossSectionParams(
    bounded_factors=(
        "ADX_14D",
        "CMF_20D",
        "CORRELATION_TO_MARKET_20D",
        "PRICE_POSITION_20D",
        "PRICE_POSITION_120D",
        "PV_CORR_20D",
        "RSI_14",
    ),
)

# v4.1: 回退到 v3.4 的 bounded_factors (4个), 与 cross_section_processor.py 保持一致
_V4_1_CROSS_SECTION = FrozenCrossSectionParams(
    bounded_factors=(
        "PRICE_POSITION_20D",
        "PRICE_POSITION_120D",
        "PV_CORR_20D",
        "RSI_14",
    ),
)

_V4_0_CONFIG = FrozenProductionConfig(
    version="v4.0",
    config_sha256=None,
    backtest=FrozenBacktestParams(),
    timing=FrozenTimingParams(),
    regime_gate=FrozenRegimeGateParams(),
    risk_control=FrozenRiskControlParams(),
    wfo=FrozenWFOParams(),
    scoring=FrozenScoringParams(),
    cross_section=_V4_0_CROSS_SECTION,
    etf_pool=FrozenETFPool(),
    strategies=(
        FrozenStrategy(
            name="strategy_1",
            factors=("ADX_14D", "OBV_SLOPE_10D", "SHARPE_RATIO_20D", "SLOPE_20D"),
        ),
        FrozenStrategy(
            name="strategy_2",
            factors=(
                "ADX_14D",
                "OBV_SLOPE_10D",
                "PRICE_POSITION_120D",
                "SHARPE_RATIO_20D",
                "SLOPE_20D",
            ),
        ),
    ),
)

_V4_1_CONFIG = FrozenProductionConfig(
    version="v4.1",
    config_sha256=None,
    backtest=FrozenBacktestParams(),
    timing=FrozenTimingParams(),
    regime_gate=FrozenRegimeGateParams(enabled=True),  # v3.4 兼容: Gate ON
    risk_control=FrozenRiskControlParams(),
    wfo=FrozenWFOParams(),
    scoring=FrozenScoringParams(),
    cross_section=_V4_1_CROSS_SECTION,
    etf_pool=FrozenETFPool(),
    strategies=(),  # 正交化后新候选待定
)

_VERSION_REGISTRY: Dict[str, FrozenProductionConfig] = {
    "v3.4": _V3_4_CONFIG,
    "v4.0": _V4_0_CONFIG,
    "v4.1": _V4_1_CONFIG,
}

CURRENT_VERSION = "v4.1"

# 操作性参数 (不校验)
_OPERATIONAL_KEYS = frozenset(
    {
        "data_dir",
        "cache_dir",
        "start_date",
        "end_date",
        "training_end_date",
        "n_jobs",
        "verbose",
        "max_workers",
        "save_all_results",
        "output_root",
        "cost_model",
        "universe",
    }
)


# ---------------------------------------------------------------------------
# Universe mode helpers (A_SHARE_ONLY / GLOBAL)
# ---------------------------------------------------------------------------


def get_universe_mode(config: Dict[str, Any]) -> str:
    """读取 universe.mode，默认 GLOBAL（向后兼容）"""
    return config.get("universe", {}).get("mode", "GLOBAL")


def get_qdii_tickers(config: Dict[str, Any]) -> set:
    """从配置读取 QDII ticker 集合，fallback 到 FrozenETFPool 硬编码"""
    raw = config.get("universe", {}).get("qdii_tickers", None)
    if raw:
        return set(str(s) for s in raw)
    return set(FrozenETFPool().qdii_codes)


def get_tradable_symbols(config: Dict[str, Any]) -> List[str]:
    """根据 universe.mode 返回可交易 ETF 列表。

    A_SHARE_ONLY → 剔除 QDII；GLOBAL → 全池。
    保持原始顺序。
    """
    all_symbols = [str(s) for s in config.get("data", {}).get("symbols", [])]
    if get_universe_mode(config) == "A_SHARE_ONLY":
        qdii = get_qdii_tickers(config)
        return [s for s in all_symbols if s not in qdii]
    return all_symbols


def is_qdii(ticker: str, config: Dict[str, Any]) -> bool:
    """判断 ticker 是否属于 QDII"""
    return ticker in get_qdii_tickers(config)


# ---------------------------------------------------------------------------
# 校验辅助
# ---------------------------------------------------------------------------


def _compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_tuple(val: Any) -> Any:
    """将 list 递归转换为 tuple 以便比较"""
    if isinstance(val, list):
        return tuple(_to_tuple(v) for v in val)
    return val


def _floats_close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9)


def _compare_values(frozen_val: Any, config_val: Any, path: str) -> List[str]:
    """比较冻结值和配置值，返回违规列表"""
    violations: List[str] = []

    # 统一 list -> tuple
    config_val = _to_tuple(config_val)

    if isinstance(frozen_val, float) and isinstance(config_val, (int, float)):
        if not _floats_close(frozen_val, float(config_val)):
            violations.append(f"{path}: frozen={frozen_val}, config={config_val}")
    elif isinstance(frozen_val, tuple) and isinstance(config_val, tuple):
        if len(frozen_val) != len(config_val):
            violations.append(
                f"{path}: 长度不匹配 frozen={len(frozen_val)}, config={len(config_val)}"
            )
        else:
            for i, (fv, cv) in enumerate(zip(frozen_val, config_val)):
                violations.extend(_compare_values(fv, cv, f"{path}[{i}]"))
    elif frozen_val != config_val:
        violations.append(f"{path}: frozen={frozen_val}, config={config_val}")

    return violations


def _validate_backtest(frozen: FrozenBacktestParams, cfg: dict) -> List[str]:
    violations: List[str] = []
    mapping = {
        "freq": frozen.freq,
        "pos_size": frozen.pos_size,
        "commission_rate": frozen.commission_rate,
        "initial_capital": frozen.initial_capital,
        "lookback_window": frozen.lookback_window,
    }
    for key, frozen_val in mapping.items():
        if key in cfg:
            violations.extend(_compare_values(frozen_val, cfg[key], f"backtest.{key}"))
    return violations


def _validate_timing(frozen: FrozenTimingParams, cfg: dict) -> List[str]:
    violations: List[str] = []
    timing_cfg = cfg.get("timing", {})
    if not timing_cfg:
        return violations

    mapping = {
        "enabled": frozen.enabled,
        "type": frozen.type,
        "extreme_threshold": frozen.extreme_threshold,
        "extreme_position": frozen.extreme_position,
    }
    for key, frozen_val in mapping.items():
        if key in timing_cfg:
            violations.extend(
                _compare_values(frozen_val, timing_cfg[key], f"backtest.timing.{key}")
            )

    # Nested: index_timing / individual_timing
    idx_cfg = timing_cfg.get("index_timing", {})
    if "enabled" in idx_cfg:
        violations.extend(
            _compare_values(
                frozen.index_timing_enabled,
                idx_cfg["enabled"],
                "backtest.timing.index_timing.enabled",
            )
        )

    ind_cfg = timing_cfg.get("individual_timing", {})
    if "enabled" in ind_cfg:
        violations.extend(
            _compare_values(
                frozen.individual_timing_enabled,
                ind_cfg["enabled"],
                "backtest.timing.individual_timing.enabled",
            )
        )

    return violations


def _validate_regime_gate(frozen: FrozenRegimeGateParams, cfg: dict) -> List[str]:
    violations: List[str] = []
    rg_cfg = cfg.get("regime_gate", {})
    if not rg_cfg:
        return violations

    mapping = {
        "enabled": frozen.enabled,
        "mode": frozen.mode,
    }
    for key, frozen_val in mapping.items():
        if key in rg_cfg:
            violations.extend(
                _compare_values(frozen_val, rg_cfg[key], f"backtest.regime_gate.{key}")
            )

    vol_cfg = rg_cfg.get("volatility", {})
    vol_mapping = {
        "proxy_symbol": frozen.proxy_symbol,
        "window": frozen.window,
        "shift_days": frozen.shift_days,
        "thresholds_pct": frozen.thresholds_pct,
        "exposures": frozen.exposures,
    }
    for key, frozen_val in vol_mapping.items():
        if key in vol_cfg:
            violations.extend(
                _compare_values(
                    frozen_val,
                    vol_cfg[key],
                    f"backtest.regime_gate.volatility.{key}",
                )
            )

    return violations


def _validate_risk_control(frozen: FrozenRiskControlParams, cfg: dict) -> List[str]:
    violations: List[str] = []
    rc_cfg = cfg.get("risk_control", {})
    if not rc_cfg:
        return violations

    mapping = {
        "enabled": frozen.enabled,
        "leverage_cap": frozen.leverage_cap,
        "stop_check_on_rebalance_only": frozen.stop_check_on_rebalance_only,
        "stop_method": frozen.stop_method,
        "trailing_stop_pct": frozen.trailing_stop_pct,
        "atr_window": frozen.atr_window,
        "atr_multiplier": frozen.atr_multiplier,
        "cooldown_days": frozen.cooldown_days,
        "etf_stop_loss": frozen.etf_stop_loss,
    }
    for key, frozen_val in mapping.items():
        if key in rc_cfg:
            violations.extend(
                _compare_values(frozen_val, rc_cfg[key], f"backtest.risk_control.{key}")
            )

    return violations


def _validate_wfo(frozen: FrozenWFOParams, cfg: dict) -> List[str]:
    violations: List[str] = []
    wfo_cfg = cfg.get("combo_wfo", {})
    if not wfo_cfg:
        return violations

    mapping = {
        "combo_sizes": frozen.combo_sizes,
        "enable_fdr": frozen.enable_fdr,
        "fdr_alpha": frozen.fdr_alpha,
        "fdr_method": frozen.fdr_method,
        "is_period": frozen.is_period,
        "oos_period": frozen.oos_period,
        "step_size": frozen.step_size,
        "top_n": frozen.top_n,
        "rebalance_frequencies": frozen.rebalance_frequencies,
    }
    for key, frozen_val in mapping.items():
        if key in wfo_cfg:
            violations.extend(
                _compare_values(frozen_val, wfo_cfg[key], f"combo_wfo.{key}")
            )

    return violations


def _validate_scoring(frozen: FrozenScoringParams, cfg: dict) -> List[str]:
    violations: List[str] = []
    scoring_cfg = cfg.get("combo_wfo", {}).get("scoring", {})
    if not scoring_cfg:
        return violations

    if "complexity_penalty_lambda" in scoring_cfg:
        violations.extend(
            _compare_values(
                frozen.complexity_penalty_lambda,
                scoring_cfg["complexity_penalty_lambda"],
                "combo_wfo.scoring.complexity_penalty_lambda",
            )
        )
    if "use_robust_scoring" in scoring_cfg:
        violations.extend(
            _compare_values(
                frozen.use_robust_scoring,
                scoring_cfg["use_robust_scoring"],
                "combo_wfo.scoring.use_robust_scoring",
            )
        )

    # weights: frozen stores as sorted tuple of (key, val), config has dict
    weights_cfg = scoring_cfg.get("vec_score_weights", {})
    if weights_cfg:
        frozen_dict = dict(frozen.weights)
        for wk, wv in weights_cfg.items():
            if wk in frozen_dict:
                violations.extend(
                    _compare_values(
                        frozen_dict[wk],
                        wv,
                        f"combo_wfo.scoring.vec_score_weights.{wk}",
                    )
                )
            else:
                violations.append(
                    f"combo_wfo.scoring.vec_score_weights.{wk}: " f"不在冻结参数中"
                )
        # Check for missing keys in config
        for wk in frozen_dict:
            if wk not in weights_cfg:
                violations.append(
                    f"combo_wfo.scoring.vec_score_weights.{wk}: "
                    f"冻结参数中存在但配置中缺失"
                )

    return violations


def _validate_cross_section(frozen: FrozenCrossSectionParams, cfg: dict) -> List[str]:
    violations: List[str] = []
    cs_cfg = cfg.get("cross_section", {})
    if not cs_cfg:
        return violations

    if "bounded_factors" in cs_cfg:
        # Compare as sets — order is not semantically meaningful
        frozen_set = set(frozen.bounded_factors)
        config_set = set(cs_cfg["bounded_factors"])
        if frozen_set != config_set:
            missing = frozen_set - config_set
            extra = config_set - frozen_set
            if missing:
                violations.append(
                    f"cross_section.bounded_factors: 缺少 {sorted(missing)}"
                )
            if extra:
                violations.append(
                    f"cross_section.bounded_factors: 多出 {sorted(extra)}"
                )
    if "winsorize_lower" in cs_cfg:
        violations.extend(
            _compare_values(
                frozen.winsorize_lower,
                cs_cfg["winsorize_lower"],
                "cross_section.winsorize_lower",
            )
        )
    if "winsorize_upper" in cs_cfg:
        violations.extend(
            _compare_values(
                frozen.winsorize_upper,
                cs_cfg["winsorize_upper"],
                "cross_section.winsorize_upper",
            )
        )

    return violations


def _validate_etf_pool(frozen: FrozenETFPool, cfg: dict) -> List[str]:
    violations: List[str] = []
    symbols_cfg = cfg.get("data", {}).get("symbols", [])
    if not symbols_cfg:
        return violations

    frozen_symbols = sorted(frozen.symbols)
    config_symbols = sorted(str(s) for s in symbols_cfg)

    if frozen_symbols != config_symbols:
        frozen_set = set(frozen_symbols)
        config_set = set(config_symbols)
        missing = frozen_set - config_set
        extra = config_set - frozen_set

        if missing:
            violations.append(
                f"data.symbols: 缺少 {len(missing)} 个ETF: {sorted(missing)}"
            )
        if extra:
            violations.append(f"data.symbols: 多出 {len(extra)} 个ETF: {sorted(extra)}")

        # 特别检查 QDII
        qdii_set = set(frozen.qdii_codes)
        missing_qdii = qdii_set - config_set
        if missing_qdii:
            violations.append(
                f"data.symbols: 缺少 QDII ETF (严重): {sorted(missing_qdii)}"
            )

    return violations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_frozen_config(
    raw_config: dict,
    config_path: Optional[str] = None,
    version: str = CURRENT_VERSION,
    strictness: Optional[StrictnessMode] = None,
) -> FrozenProductionConfig:
    """
    加载冻结配置并校验 raw_config 是否与冻结值一致。

    Parameters
    ----------
    raw_config : dict
        从 YAML 加载的配置字典
    config_path : str, optional
        配置文件路径 (用于 SHA256 校验)
    version : str
        冻结参数版本 (默认: CURRENT_VERSION)
    strictness : StrictnessMode, optional
        校验严格度。默认从环境变量 FROZEN_PARAMS_MODE 读取，
        未设置则为 STRICT。

    Returns
    -------
    FrozenProductionConfig

    Raises
    ------
    FrozenParamViolation
        STRICT 模式下发现参数不一致
    KeyError
        未知版本号
    """
    if version not in _VERSION_REGISTRY:
        raise KeyError(f"未知的冻结参数版本: {version}")

    frozen = _VERSION_REGISTRY[version]

    # 确定 strictness
    if strictness is None:
        env_mode = os.environ.get("FROZEN_PARAMS_MODE", "").lower()
        if env_mode == "warn":
            strictness = StrictnessMode.WARN
        else:
            strictness = StrictnessMode.STRICT

    # SHA256 快速路径
    config_sha256: Optional[str] = None
    if config_path:
        p = Path(config_path)
        if p.exists():
            config_sha256 = _compute_file_sha256(p)

    # 逐字段校验
    backtest_cfg = raw_config.get("backtest", {})
    violations: List[str] = []
    violations.extend(_validate_backtest(frozen.backtest, backtest_cfg))
    violations.extend(_validate_timing(frozen.timing, backtest_cfg))
    violations.extend(_validate_regime_gate(frozen.regime_gate, backtest_cfg))
    violations.extend(_validate_risk_control(frozen.risk_control, backtest_cfg))
    violations.extend(_validate_wfo(frozen.wfo, raw_config))
    violations.extend(_validate_scoring(frozen.scoring, raw_config))
    violations.extend(_validate_cross_section(frozen.cross_section, raw_config))
    violations.extend(_validate_etf_pool(frozen.etf_pool, raw_config))

    if violations:
        if strictness == StrictnessMode.STRICT:
            raise FrozenParamViolation(violations)
        else:
            for v in violations:
                logger.warning(f"参数冻结违规 (WARN模式): {v}")

    # 返回带 SHA256 的 frozen config
    return FrozenProductionConfig(
        version=frozen.version,
        config_sha256=config_sha256,
        backtest=frozen.backtest,
        timing=frozen.timing,
        regime_gate=frozen.regime_gate,
        risk_control=frozen.risk_control,
        wfo=frozen.wfo,
        scoring=frozen.scoring,
        cross_section=frozen.cross_section,
        etf_pool=frozen.etf_pool,
        strategies=frozen.strategies,
    )
