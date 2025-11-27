#!/usr/bin/env python3
"""Compatibility shim: delegates to strategies.backtest.production_backtest."""

from strategies.backtest.production_backtest import *  # noqa: F401,F403

if __name__ == "__main__":
    from strategies.backtest.production_backtest import main

    main()
