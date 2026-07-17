# Trading bot data directory

Runtime artifacts land here:

- `KILL_SWITCH`   — created by operator to halt the bot
- `heartbeat`     — timestamp of last successful cycle (watchdog)
- `state.json`    — persistable risk state for crash recovery
- `equity_<SYMBOL>.csv` — backtest equity curve
- `trades_<SYMBOL>.csv` — backtest trade log

Safe to delete between runs (the bot will recreate as needed).
