"""architecture/memory_system.py
=====================================================================
Memory System (Improvement #11)
=====================================================================
Gives the bot long-term memory of past trades, market conditions, and
outcomes so it can learn from experience without full retraining.

Three Memory Types:

1. EPISODIC MEMORY — specific trade events
   "On 2024-03-15 14:30, I bought BTCUSD at 41250, ATR was 850,
    RSI was 62, regime was 'trend_up'. Position hit TP at 43300 in 4h.
    PnL: +$420. Lesson: trend-continuation in low-vol regime works."

2. SEMANTIC MEMORY — generalized patterns
   "When BTC.D > 55% AND altcoin correlation > 0.8, altcoin rallies
    tend to fail. Win rate of long entries in this state: 28%."

3. PROCEDURAL MEMORY — strategy knowledge
   "Momentum strategy on M15 BTCUSD: optimal when ADX > 25 and
    volume > 1.5× average. Stops at 1.5×ATR. TP at 2.5×ATR."

Memory Operations:
    - ENCODE: convert a trade outcome into a memory
    - RETRIEVE: find similar past situations (k-NN on feature vectors)
    - CONSOLIDATE: nightly job that merges episodic → semantic
    - FORGET: discard outdated memories (decay function)
    - RECALL: query memories by symbol, regime, outcome, similarity

Usage:
    memory = MemorySystem(db_path="data/trading_bot.db")
    memory.encode_episode(trade_outcome)
    similar = memory.retrieve_similar(current_features, top_k=5)
    expected_value = memory.estimate_ev(current_features, signal)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger("trading_bot.architecture.memory_system")


@dataclass
class EpisodicMemory:
    """A single trade episode."""
    id: str = ""
    timestamp: str = ""
    symbol: str = ""
    timeframe: str = ""
    direction: str = ""  # BUY / SELL
    # Entry context
    features: Dict[str, Any] = field(default_factory=dict)
    regime: str = ""
    # Trade details
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    lots: float = 0.0
    hold_time_s: float = 0.0
    # Outcome
    pnl: float = 0.0
    outcome: str = ""  # win, loss, breakeven
    r_multiple: float = 0.0  # pnl / initial_risk
    # Strategy
    strategy_name: str = ""
    # Embedding for similarity search (pre-computed)
    feature_hash: str = ""


@dataclass
class SemanticMemory:
    """A generalized pattern learned from many episodes."""
    id: str = ""
    created_at: str = ""
    updated_at: str = ""
    pattern_description: str = ""
    condition_features: Dict[str, Any] = field(default_factory=dict)
    sample_count: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    avg_r_multiple: float = 0.0
    confidence: float = 0.0  # 0-1, based on sample size


class MemorySystem:
    """Long-term memory of trades and patterns.

    Uses SQLite for persistence + in-memory index for fast retrieval.
    Similarity search uses cosine similarity on feature vectors.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS episodic_memory (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        symbol TEXT,
        timeframe TEXT,
        direction TEXT,
        features TEXT,
        regime TEXT,
        entry_price REAL,
        exit_price REAL,
        sl REAL,
        tp REAL,
        lots REAL,
        hold_time_s REAL,
        pnl REAL,
        outcome TEXT,
        r_multiple REAL,
        strategy_name TEXT,
        feature_hash TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_ep_symbol ON episodic_memory(symbol);
    CREATE INDEX IF NOT EXISTS idx_ep_outcome ON episodic_memory(outcome);
    CREATE INDEX IF NOT EXISTS idx_ep_regime ON episodic_memory(regime);

    CREATE TABLE IF NOT EXISTS semantic_memory (
        id TEXT PRIMARY KEY,
        created_at TEXT,
        updated_at TEXT,
        pattern_description TEXT,
        condition_features TEXT,
        sample_count INTEGER,
        win_rate REAL,
        avg_pnl REAL,
        avg_r_multiple REAL,
        confidence REAL
    );
    """

    def __init__(self,
                 db_path: str = "data/trading_bot.db",
                 decay_half_life_days: float = 90.0):
        self._lock = threading.RLock()
        self._db_path = db_path
        self._decay_half_life = decay_half_life_days
        self._episodes: List[EpisodicMemory] = []
        self._semantics: List[SemanticMemory] = []
        self._init_db()
        self._load_into_memory()

    def _init_db(self) -> None:
        # Co-Founder Audit Fix: leak-safe connection. Previous pattern
        # leaked the connection on any exception between connect and close.
        conn = None
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.executescript(self.SCHEMA)
            conn.commit()
        except Exception as e:  # noqa: BLE001
            log.warning("memory: DB init failed: %r", e)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _load_into_memory(self) -> None:
        """C7 fix: load only the most recent 1000 episodes (not all 5000)
        into RAM. For similarity search, recent episodes are more relevant
        than ancient ones (market regime drift makes old episodes less
        useful). The DB retains all episodes for compliance; this just
        limits the in-memory working set to avoid OOM on long-running bots.
        """
        # Co-Founder Audit Fix: leak-safe connection.
        conn = None
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            # C7 fix: LIMIT 1000 instead of 5000 to bound memory usage.
            rows = conn.execute(
                "SELECT * FROM episodic_memory ORDER BY timestamp DESC LIMIT 1000"
            ).fetchall()
            for r in rows:
                ep = EpisodicMemory(
                    id=r["id"],
                    timestamp=r["timestamp"],
                    symbol=r["symbol"],
                    timeframe=r["timeframe"],
                    direction=r["direction"],
                    features=json.loads(r["features"] or "{}"),
                    regime=r["regime"],
                    entry_price=r["entry_price"],
                    exit_price=r["exit_price"],
                    sl=r["sl"],
                    tp=r["tp"],
                    lots=r["lots"],
                    hold_time_s=r["hold_time_s"],
                    pnl=r["pnl"],
                    outcome=r["outcome"],
                    r_multiple=r["r_multiple"],
                    strategy_name=r["strategy_name"],
                    feature_hash=r["feature_hash"],
                )
                self._episodes.append(ep)
            log.info("memory: loaded %d episodes from DB (capped at 1000 for RAM safety)",
                      len(self._episodes))
        except Exception as e:  # noqa: BLE001
            log.warning("memory: load failed: %r", e)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # ENCODE — convert trade outcome into memory
    # ------------------------------------------------------------------
    def encode_episode(self,
                       symbol: str,
                       timeframe: str,
                       direction: str,
                       features: Dict[str, Any],
                       regime: str,
                       entry_price: float,
                       exit_price: float,
                       sl: float,
                       tp: float,
                       lots: float,
                       hold_time_s: float,
                       pnl: float,
                       strategy_name: str = "") -> str:
        """Encode a completed trade as an episodic memory."""
        risk = abs(entry_price - sl) * lots
        r_mult = pnl / risk if risk > 0 else 0.0
        outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")

        ep = EpisodicMemory(
            id=f"ep_{int(time.time()*1000)}",
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            features=features,
            regime=regime,
            entry_price=entry_price,
            exit_price=exit_price,
            sl=sl, tp=tp, lots=lots,
            hold_time_s=hold_time_s,
            pnl=pnl,
            outcome=outcome,
            r_multiple=r_mult,
            strategy_name=strategy_name,
            feature_hash=self._hash_features(features),
        )
        with self._lock:
            self._episodes.append(ep)
            # C7 fix: in-memory cap reduced from 5000 to 1000 to bound RAM.
            # Old episodes remain in the DB for compliance/audit; this just
            # limits the in-memory working set for similarity search.
            if len(self._episodes) > 1000:
                self._episodes = self._episodes[-1000:]
        self._persist_episode(ep)
        log.info("memory: encoded episode %s (%s %s pnl=%.2f R=%.2f)",
                 ep.id, symbol, direction, pnl, r_mult)
        return ep.id

    # ------------------------------------------------------------------
    # RETRIEVE — find similar past situations
    # ------------------------------------------------------------------
    def retrieve_similar(self,
                         current_features: Dict[str, Any],
                         symbol: Optional[str] = None,
                         regime: Optional[str] = None,
                         top_k: int = 10) -> List[Tuple[float, EpisodicMemory]]:
        """Find the k most similar past episodes (cosine similarity)."""
        if not self._episodes:
            return []
        # Convert current features to vector
        keys = sorted(set(current_features.keys()) |
                     {k for ep in self._episodes for k in ep.features.keys()})
        curr_vec = self._to_vector(current_features, keys)
        curr_norm = np.linalg.norm(curr_vec)
        if curr_norm == 0:
            return []

        scored = []
        with self._lock:
            for ep in self._episodes:
                if symbol and ep.symbol != symbol:
                    continue
                if regime and ep.regime != regime:
                    continue
                ep_vec = self._to_vector(ep.features, keys)
                ep_norm = np.linalg.norm(ep_vec)
                if ep_norm == 0:
                    continue
                sim = float(np.dot(curr_vec, ep_vec) / (curr_norm * ep_norm))
                # Apply time decay (older memories count less)
                age_days = (time.time() - _parse_iso(ep.timestamp)) / 86400
                decay = 0.5 ** (age_days / self._decay_half_life)
                sim *= decay
                scored.append((sim, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def estimate_ev(self,
                    current_features: Dict[str, Any],
                    symbol: Optional[str] = None,
                    regime: Optional[str] = None,
                    top_k: int = 20) -> Dict[str, float]:
        """Estimate expected value from similar past episodes.

        MEMORY SYSTEM AUDIT FIX: weight each episode's contribution by its
        similarity score. Previously all similar episodes counted equally —
        a barely-similar episode (sim=0.01) had the same weight as a very
        similar one (sim=0.95). Now the EV is a similarity-weighted average.
        """
        similar = self.retrieve_similar(current_features, symbol, regime, top_k)
        if not similar:
            return {"ev": 0.0, "win_rate": 0.5, "sample_size": 0,
                   "avg_pnl": 0.0, "avg_r": 0.0}
        # MEMORY SYSTEM AUDIT FIX: weight by similarity score.
        # sim_weight = similarity / sum(similarities) — so episodes with
        # higher similarity contribute more to the EV estimate.
        total_weight = sum(max(s, 0.001) for s, _ in similar)  # floor at 0.001
        wins = sum(max(s, 0.001) for s, ep in similar if ep.outcome == "win")
        losses = sum(max(s, 0.001) for s, ep in similar if ep.outcome == "loss")
        total_pnl = sum(ep.pnl * max(s, 0.001) for s, ep in similar) / total_weight
        avg_r = sum(ep.r_multiple * max(s, 0.001) for s, ep in similar) / total_weight
        win_rate = wins / max(wins + losses, 1e-9)
        ev = (win_rate * avg_r) - ((1 - win_rate) * 1.0)  # in R multiples
        return {
            "ev": ev,
            "win_rate": win_rate,
            "sample_size": len(similar),
            "avg_pnl": total_pnl,
            "avg_r": avg_r,
        }

    # ------------------------------------------------------------------
    # CONSOLIDATE — nightly job: merge episodes → semantic memories
    # ------------------------------------------------------------------
    def consolidate(self) -> int:
        """Mine episodic memories for patterns and create semantic memories.

        H3/X5 fix: this method should be called periodically by the main
        loop (e.g. every 100 cycles). The caller is responsible for
        scheduling — see TradingBot.cycle() for the integration point.
        """
        # Group by regime + outcome
        groups: Dict[Tuple[str, str], List[EpisodicMemory]] = {}
        with self._lock:
            for ep in self._episodes:
                key = (ep.regime or "unknown", ep.direction)
                groups.setdefault(key, []).append(ep)

        new_semantics = 0
        for (regime, direction), eps in groups.items():
            if len(eps) < 5:
                continue  # need at least 5 samples
            wins = sum(1 for e in eps if e.outcome == "win")
            wr = wins / len(eps)
            avg_pnl = sum(e.pnl for e in eps) / len(eps)
            avg_r = sum(e.r_multiple for e in eps) / len(eps)
            confidence = min(1.0, len(eps) / 30.0)

            # H3 fix: store a summary of the feature conditions that define
            # this pattern, not just regime+direction, so semantic memory
            # is actually useful for retrieval.
            feature_summary: Dict[str, float] = {}
            for e in eps:
                for k, v in e.features.items():
                    try:
                        feature_summary[k] = feature_summary.get(k, 0.0) + float(v)
                    except (TypeError, ValueError):
                        continue
            for k in list(feature_summary.keys()):
                feature_summary[k] /= len(eps)

            sm = SemanticMemory(
                id=f"sm_{regime}_{direction}_{int(time.time())}",
                created_at=datetime.now(tz=timezone.utc).isoformat(),
                updated_at=datetime.now(tz=timezone.utc).isoformat(),
                pattern_description=f"Regime={regime}, Direction={direction}: "
                                    f"win_rate={wr:.1%}, avg_R={avg_r:.2f}",
                condition_features={"regime": regime, "direction": direction,
                                    "avg_features": feature_summary},
                sample_count=len(eps),
                win_rate=wr,
                avg_pnl=avg_pnl,
                avg_r_multiple=avg_r,
                confidence=confidence,
            )
            with self._lock:
                self._semantics.append(sm)
            self._persist_semantic(sm)
            new_semantics += 1

        log.info("memory: consolidated %d semantic memories", new_semantics)
        return new_semantics

    def prune_old_episodes(self, max_age_days: int = 365) -> int:
        """Delete episodes older than `max_age_days` from the DB and RAM.

        Fixes the 'no memory expiration' issue — without this, the
        episodic_memory table grows indefinitely. Default 365 days
        retains a full year of history for compliance while bounding
        DB size. Returns the number of episodes pruned.
        """
        # Co-Founder Audit Fix: leak-safe connection.
        conn = None
        pruned = 0
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            cursor = conn.execute(
                "DELETE FROM episodic_memory "
                "WHERE timestamp < datetime('now', ?)",
                (f"-{max_age_days} days",))
            pruned = cursor.rowcount
            conn.commit()
        except Exception as e:  # noqa: BLE001
            log.warning("memory: prune failed: %r", e)
            return 0
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        # Also prune from RAM
        with self._lock:
            before = len(self._episodes)
            # P0-10 fix: compare ISO timestamps, not string-to-float.
            cutoff_ts = (datetime.now(tz=timezone.utc).timestamp() - max_age_days * 86400)
            from datetime import datetime as _dt
            self._episodes = [e for e in self._episodes
                              if _dt.fromisoformat(e.timestamp).timestamp() > cutoff_ts]
            log.info("memory: pruned %d old episodes (DB=%d, RAM=%d)",
                     pruned, pruned, before - len(self._episodes))
        return pruned

    # ------------------------------------------------------------------
    # RECALL — query semantic memories
    # ------------------------------------------------------------------
    def recall_patterns(self,
                        regime: Optional[str] = None,
                        min_confidence: float = 0.3) -> List[Dict[str, Any]]:
        with self._lock:
            out = []
            for sm in self._semantics:
                if sm.confidence < min_confidence:
                    continue
                if regime and sm.condition_features.get("regime") != regime:
                    continue
                out.append(asdict(sm))
            return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_vector(self, features: Dict[str, Any],
                   keys: List[str]) -> np.ndarray:
        out = []
        for k in keys:
            v = features.get(k, 0)
            try:
                out.append(float(v) if v is not None else 0.0)
            except (TypeError, ValueError):
                out.append(0.0)
        return np.array(out, dtype=float)

    def _hash_features(self, features: Dict[str, Any]) -> str:
        # M6 fix: use SHA-256 instead of MD5 to eliminate collision risk.
        # MD5 truncated to 12 hex chars has a non-trivial collision probability
        # across thousands of episodes; SHA-256 truncated to 16 is much safer.
        import hashlib
        s = ",".join(f"{k}={features[k]}" for k in sorted(features.keys()))
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def _persist_episode(self, ep: EpisodicMemory) -> None:
        # Co-Founder Audit Fix: leak-safe connection.
        conn = None
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.execute("""
                INSERT OR REPLACE INTO episodic_memory
                (id, timestamp, symbol, timeframe, direction, features, regime,
                 entry_price, exit_price, sl, tp, lots, hold_time_s, pnl,
                 outcome, r_multiple, strategy_name, feature_hash)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ep.id, ep.timestamp, ep.symbol, ep.timeframe, ep.direction,
                json.dumps(ep.features, default=str), ep.regime,
                ep.entry_price, ep.exit_price, ep.sl, ep.tp, ep.lots,
                ep.hold_time_s, ep.pnl, ep.outcome, ep.r_multiple,
                ep.strategy_name, ep.feature_hash,
            ))
            conn.commit()
        except Exception as e:  # noqa: BLE001
            log.warning("memory: persist episode failed: %r", e)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _persist_semantic(self, sm: SemanticMemory) -> None:
        # Co-Founder Audit Fix: leak-safe connection.
        conn = None
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.execute("""
                INSERT OR REPLACE INTO semantic_memory
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                sm.id, sm.created_at, sm.updated_at, sm.pattern_description,
                json.dumps(sm.condition_features, default=str),
                sm.sample_count, sm.win_rate, sm.avg_pnl,
                sm.avg_r_multiple, sm.confidence,
            ))
            conn.commit()
        except Exception as e:  # noqa: BLE001
            log.warning("memory: persist semantic failed: %r", e)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "episodic_count": len(self._episodes),
                "semantic_count": len(self._semantics),
                "symbols_in_memory": list({e.symbol for e in self._episodes}),
                "regimes_in_memory": list({e.regime for e in self._episodes}),
            }


def _parse_iso(s: str) -> float:
    """Parse ISO timestamp to epoch seconds.

    MEMORY SYSTEM AUDIT FIX: on parse failure, return 0.0 (oldest possible)
    instead of time.time() (now). Previously a malformed timestamp made an
    episode appear "just now" — no time decay applied — so stale/corrupt
    memories got the highest weight in similarity search. Now they get the
    lowest weight (maximum decay).
    """
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0
