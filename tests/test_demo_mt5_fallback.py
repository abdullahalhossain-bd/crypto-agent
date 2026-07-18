import types
import unittest

from architecture.circuit_breaker import BrokerDisconnectBreaker, CircuitBreakerCoordinator
from architecture.integration import TradingBot


class DummyExchange:
    def __init__(self, name: str, connect_ok: bool = True):
        self.name = name
        self.connected = False
        self._connect_ok = connect_ok

    def connect(self) -> bool:
        self.connected = self._connect_ok
        return self._connect_ok

    def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def account_info(self):
        return types.SimpleNamespace(login=0, server="", balance=10000.0, equity=10000.0, leverage=100)

    def fetch_candles(self, *args, **kwargs):
        return []

    def symbol_tick(self, symbol: str):
        return None

    def symbol_info(self, symbol: str):
        return None

    def get_symbols_by_pattern(self, patterns):
        return []

    def positions(self, symbol=None):
        return []

    def place_order(self, req):
        return None

    def modify_order(self, ticket, sl, tp):
        return None

    def close_order(self, ticket, symbol, volume, side):
        return None

    def pending_orders(self):
        return []

    def order_history(self, limit=100):
        return []


class DemoFallbackTests(unittest.TestCase):
    def test_demo_mode_ignores_broker_disconnect_breaker(self):
        coordinator = CircuitBreakerCoordinator(
            {"circuit_breakers": {}},
            ignore_broker_disconnect=True,
        )
        self.assertFalse(any(isinstance(b, BrokerDisconnectBreaker) for b in coordinator.breakers))

    def test_demo_mode_falls_back_to_paper_when_mt5_connect_fails(self):
        created = []

        def fake_create_exchange(exchange_type: str, **kwargs):
            created.append(exchange_type)
            if exchange_type == "mt5":
                return DummyExchange("mt5", connect_ok=False)
            if exchange_type == "paper":
                return DummyExchange("paper", connect_ok=True)
            raise AssertionError(f"unexpected exchange type: {exchange_type}")

        import architecture.integration as integration_module
        original = integration_module.create_exchange
        integration_module.create_exchange = fake_create_exchange
        try:
            bot = TradingBot({
                "mode": "demo",
                "capital": 10000.0,
                "symbols": [{"name": "BTCUSD"}],
                "symbols_auto_load": False,
                "mt5": {"login": 123456, "password": "pw", "server": "MetaQuotes-Demo"},
                "runtime": {"poll_interval_s": 1.0, "heartbeat_timeout_s": 10.0, "kill_switch_file": "data/KILL_SWITCH"},
                "risk": {"max_spread_bps": 10.0, "max_daily_loss": 0.05, "risk_per_trade": 0.01},
                "strategy": {},
                "execution": {"execution_mode": "paper"},
            }, mode="demo")

            bot._load_symbols = lambda: ["BTCUSD"]
            bot._register_recoveries = lambda: None
            bot.feature_pipeline.warmup_requirement = lambda: 0

            ok = bot.boot()

            self.assertTrue(ok)
            self.assertEqual(created[0], "mt5")
            self.assertEqual(created[1], "paper")
            self.assertEqual(bot.exchange.name, "paper")
            self.assertTrue(bot.exchange.is_connected())
        finally:
            integration_module.create_exchange = original


if __name__ == "__main__":
    unittest.main()
