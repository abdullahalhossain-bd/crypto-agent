"""external package — Phase 9 External Integrations (Day 161-180).

Integrates the trading bot with external services:
  - env_loader         : loads .env file + provides typed access
  - llm_provider       : multi-key LLM with fallback chain
  - market_data        : Twelve Data / Alpha Vantage / Polygon adapters
  - news_provider      : FRED / NewsAPI / Forex Factory economic calendar
  - sentiment_provider : Myfxbook / OANDA retail sentiment
  - polymarket         : prediction market data (no key needed)
"""
from external.env_loader import EnvLoader  # noqa: F401
from external.llm_provider import (  # noqa: F401
    LLMProvider, LLMMessage, LLMResponse, LLMProviderError,
)
from external.market_data import (  # noqa: F401
    MarketDataManager, MarketDataResult,
)
from external.news_provider import (  # noqa: F401
    NewsProviderManager, EconomicEvent,
)
from external.sentiment_provider import (  # noqa: F401
    SentimentProviderManager, SentimentData,
)
from external.polymarket import (  # noqa: F401
    get_prediction_markets, get_crypto_markets, format_markets_for_prompt,
)

__all__ = [
    "EnvLoader",
    "LLMProvider", "LLMMessage", "LLMResponse", "LLMProviderError",
    "MarketDataManager", "MarketDataResult",
    "NewsProviderManager", "EconomicEvent",
    "SentimentProviderManager", "SentimentData",
    "get_prediction_markets", "get_crypto_markets", "format_markets_for_prompt",
]
