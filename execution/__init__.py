"""execution package — alpha-aware execution layer.

Phase 6: AlphaExecutionEngine was archived to legacy/ (it was a Stack-B
wrapper over the now-archived engine.execution.ExecutionEngine). This
package now exposes only the slicer and slippage model, which are wired
into TradingBot._place_order_with_slicing.
"""
from execution.slippage_model import SlippageModel  # noqa: F401
from execution.order_slicer import OrderSlicer, SlicedOrder  # noqa: F401

__all__ = ["SlippageModel", "OrderSlicer", "SlicedOrder"]
