"""observability.dashboard
=====================================================================
Day 29 — Dashboard renderer.

Renders a self-contained HTML file from the latest metrics snapshot
+ recent decision traces. No JavaScript framework required — just
plain HTML + CSS + a couple of inline SVG sparklines. Operators can
open it directly from disk (`file://` URL) without a server.

Refresh strategy: the main loop calls `render()` every N cycles; the
HTML file is overwritten in place. A browser tab pointed at the file
can auto-refresh with a meta refresh tag (we embed one).
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Optional

from observability.decision_trace import DecisionTraceRecorder
from observability.metrics import MetricsSnapshot
from utils.logger import get_logger

log = get_logger("observability.dashboard")


class DashboardRenderer:
    def __init__(self, output_path: str = "data/dashboard.html",
                 auto_refresh_s: int = 30) -> None:
        self.output_path = output_path
        self.auto_refresh_s = int(auto_refresh_s)
        self.trace_recorder = DecisionTraceRecorder()

    # ----------------------------------------------------------------
    def render(self, snapshot: Optional[MetricsSnapshot],
               extra: dict[str, Any] | None = None) -> str:
        extra = extra or {}
        recent_traces = self.trace_recorder.query(limit=20)
        html = self._html(snapshot, recent_traces, extra)
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(html)
        except Exception as e:  # noqa: BLE001
            log.warning("dashboard write failed: %r", e)
        return html

    # ----------------------------------------------------------------
    def _html(self, snap: Optional[MetricsSnapshot],
              traces: list[dict[str, Any]],
              extra: dict[str, Any]) -> str:
        s = snap.to_dict() if snap else {}
        rows_html = self._rows_html(s)
        traces_html = self._traces_html(traces)
        sparkline = self._sparkline_svg(extra.get("equity_curve", []))
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trading Bot Dashboard</title>
<meta http-equiv="refresh" content="{self.auto_refresh_s}">
<style>
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
          margin: 0; padding: 24px; background: #0e1117; color: #c9d1d9; }}
  h1   {{ color: #58a6ff; margin-top: 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
           gap: 16px; margin-bottom: 24px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px;
           padding: 16px; }}
  .card .label {{ color: #8b949e; font-size: 12px; text-transform: uppercase;
                  letter-spacing: 0.5px; }}
  .card .value {{ font-size: 24px; font-weight: 600; margin-top: 4px; }}
  .good  {{ color: #3fb950; }}
  .bad   {{ color: #f85149; }}
  .neutral {{ color: #c9d1d9; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px;
           background: #161b22; border-radius: 6px; overflow: hidden; }}
  th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; }}
  th     {{ background: #21262d; color: #8b949e; text-transform: uppercase;
            font-size: 11px; letter-spacing: 0.5px; }}
  .sparkline {{ width: 100%; height: 60px; margin-top: 8px; }}
  .footer {{ color: #6e7681; font-size: 11px; margin-top: 24px; }}
  .status-pill {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
                  font-size: 11px; font-weight: 600; text-transform: uppercase; }}
  .pill-approved  {{ background: rgba(63,185,80,0.2); color: #3fb950; }}
  .pill-rejected  {{ background: rgba(248,81,73,0.2); color: #f85149; }}
  .pill-executed  {{ background: rgba(88,166,255,0.2); color: #58a6ff; }}
  .pill-failed    {{ background: rgba(248,81,73,0.2); color: #f85149; }}
  .pill-pending   {{ background: rgba(139,148,158,0.2); color: #8b949e; }}
</style>
</head>
<body>
  <h1>Trading Bot Dashboard</h1>
  <div class="grid">
    {rows_html}
    <div class="card" style="grid-column: 1 / -1;">
      <div class="label">Equity curve</div>
      {sparkline}
    </div>
  </div>

  <h2>Recent decisions (last 20)</h2>
  <table>
    <thead>
      <tr><th>Time</th><th>Symbol</th><th>Action</th><th>Status</th>
          <th>Reason</th><th>Strategy</th></tr>
    </thead>
    <tbody>
      {traces_html}
    </tbody>
  </table>

  <div class="footer">
    Generated at {datetime.now(tz=timezone.utc).isoformat()} ·
    auto-refresh every {self.auto_refresh_s}s ·
    file: {self.output_path}
  </div>
</body>
</html>
"""

    # ----------------------------------------------------------------
    @staticmethod
    def _rows_html(s: dict[str, Any]) -> str:
        if not s:
            return '<div class="card"><div class="label">No data yet</div></div>'
        def fmt_money(v: float) -> str:
            try:
                return f"${float(v):,.2f}"
            except Exception:
                return str(v)
        def fmt_pct(v: float) -> str:
            try:
                return f"{float(v):.2%}"
            except Exception:
                return str(v)
        def cls_for(metric: str, v: float) -> str:
            try:
                if metric in ("realized_pnl", "unrealized_pnl"):
                    return "good" if float(v) > 0 else ("bad" if float(v) < 0 else "neutral")
                if metric == "win_rate":
                    return "good" if float(v) >= 0.5 else "neutral"
                if metric == "max_drawdown_pct":
                    return "bad" if float(v) > 0.1 else "neutral"
            except Exception:
                pass
            return "neutral"
        cards = [
            ("Equity", fmt_money(s.get("equity", 0)), cls_for("realized_pnl", s.get("equity", 0))),
            ("Realised PnL", fmt_money(s.get("realized_pnl", 0)), cls_for("realized_pnl", s.get("realized_pnl", 0))),
            ("Unrealised PnL", fmt_money(s.get("unrealized_pnl", 0)), cls_for("unrealized_pnl", s.get("unrealized_pnl", 0))),
            ("Win rate", fmt_pct(s.get("win_rate", 0)), cls_for("win_rate", s.get("win_rate", 0))),
            ("Profit factor", f"{s.get('profit_factor', 0):.2f}", "neutral"),
            ("Max drawdown", fmt_pct(s.get("max_drawdown_pct", 0)), cls_for("max_drawdown_pct", s.get("max_drawdown_pct", 0))),
            ("Sharpe", f"{s.get('sharpe', 0):.2f}", "neutral"),
            ("Sortino", f"{s.get('sortino', 0):.2f}", "neutral"),
            ("Gross exposure", f"{s.get('gross_exposure', 0):.2f}", "neutral"),
            ("Net exposure", f"{s.get('net_exposure', 0):.2f}", "neutral"),
            ("Open positions", str(s.get("n_positions", 0)), "neutral"),
            ("Total trades", str(s.get("n_open_trades_total", 0)), "neutral"),
        ]
        return "\n".join(
            f'<div class="card"><div class="label">{lbl}</div>'
            f'<div class="value {cls}">{val}</div></div>'
            for lbl, val, cls in cards
        )

    # ----------------------------------------------------------------
    @staticmethod
    def _traces_html(traces: list[dict[str, Any]]) -> str:
        if not traces:
            return ('<tr><td colspan="6" style="text-align:center;color:#6e7681;">'
                    'No decisions yet</td></tr>')
        rows = []
        for t in traces[:20]:
            ts = t.get("ts", "")[:19].replace("T", " ")
            sym = t.get("symbol", "")
            act = t.get("action", "")
            status = t.get("final_status", "pending")
            reason = t.get("final_reason", "")
            strat = ""
            sig = t.get("signal", {}) or {}
            strat = sig.get("strategy", "") or (t.get("ml_score") or {}).get("strategy", "")
            pill_cls = f"pill-{status}"
            rows.append(
                f"<tr><td>{ts}</td><td>{sym}</td><td>{act}</td>"
                f'<td><span class="status-pill {pill_cls}">{status}</span></td>'
                f"<td>{reason}</td><td>{strat}</td></tr>"
            )
        return "\n".join(rows)

    # ----------------------------------------------------------------
    @staticmethod
    def _sparkline_svg(equity_curve: list[float]) -> str:
        if len(equity_curve) < 2:
            return '<svg class="sparkline"></svg>'
        import numpy as np
        arr = np.array(equity_curve, dtype=float)
        mn, mx = float(arr.min()), float(arr.max())
        if mx == mn:
            mx = mn + 1.0
        W, H = 600, 60
        pts = []
        for i, v in enumerate(arr):
            x = (i / (len(arr) - 1)) * W
            y = H - ((v - mn) / (mx - mn)) * H
            pts.append(f"{x:.1f},{y:.1f}")
        stroke = "#3fb950" if arr[-1] >= arr[0] else "#f85149"
        return (
            f'<svg class="sparkline" viewBox="0 0 {W} {H}" '
            f'preserveAspectRatio="none">'
            f'<polyline fill="none" stroke="{stroke}" stroke-width="1.5" '
            f'points="{" ".join(pts)}"/></svg>'
        )
