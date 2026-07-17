"""Trading Bot package root.

Keeps imports stable across the codebase. Sub-packages:
  - brokers/   : MT5 connection wrapper
  - engine/    : data_feed, signals, strategy, risk, execution
  - backtest/  : historical replay engine
  - utils/     : logger, indicators
"""
__version__ = "0.7.0"
__all__ = ["__version__"]
