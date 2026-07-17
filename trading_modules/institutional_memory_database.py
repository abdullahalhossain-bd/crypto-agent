"""trading_modules/institutional_memory_database.py
=====================================================================
Institutional Memory Database (Principle #173, #174)
=====================================================================
Long-term memory of every market situation the bot has encountered.
Builds a searchable knowledge base of:
    - Market regimes + outcomes
    - Pattern library (what worked, what didn't)
    - Execution archive (how well did we execute)
    - Emotion index (market sentiment at the time)
    - Macro context (what was happening in the world)

This is the bot's "experience" — every trade adds to it, and every new
decision can query it for similar past situations.

Memory Types:
    1. PATTERN LIBRARY — chart patterns + their outcomes
    2. REGIME HISTORY — market regimes + which strategies worked
    3. OUTCOME ARCHIVE — trade results with full context
    4. EXECUTION ARCHIVE — how well orders were filled
    5. EMOTION INDEX — market sentiment snapshots
    6. MACRO CONTEXT — economic events + market impact

Storage: SQLite for persistence + in-memory index for fast retrieval
Retrieval: similarity search on feature vectors

Usage:
    db = InstitutionalMemoryDatabase()

    # Store a memory
    db.store_pattern(
        symbol="BTCUSD", pattern="bull_flag",
        features={"rsi": 62, "atr_pct": 1.5, "volume": 1.3},
        outcome="win", pnl=42, r_multiple=1.8,
        regime="trend_up", session="london",
    )

    # Query similar memories
    similar = db.query_similar(
        symbol="BTCUSD",
        features={"rsi": 60, "atr_pct": 1.4, "volume": 1.2},
        top_k=5,
    )
    # similar = [{"pattern": "bull_flag", "win_rate": 0.75, "avg_r": 1.6}, ...]
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger("trading_bot.institutional_memory_database")


@dataclass
class Memory:
    """A single memory entry."""
    id: str = ""
    timestamp: str = ""
    memory_type: str = ""         # pattern/regime/outcome/execution/emotion/macro
    symbol: str = ""
    pattern: str = ""             # pattern name (if applicable)
    # Context
    features: Dict[str, Any] = field(default_factory=dict)
    regime: str = ""
    session: str = ""
    emotion: str = ""
    macro_context: str = ""
    # Outcome
    outcome: str = ""             # win/loss/breakeven
    pnl: float = 0.0
    r_multiple: float = 0.0
    # Execution
    execution_quality: float = 0.0
    slippage_bps: float = 0.0
    # Feature hash for similarity search
    feature_hash: str = ""
    # Notes
    notes: str = ""


class InstitutionalMemoryDatabase:
    """Long-term institutional memory with similarity search.

    Uses SQLite for persistence + in-memory cache for fast retrieval.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS institutional_memory (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        memory_type TEXT,
        symbol TEXT,
        pattern TEXT,
        features TEXT,
        regime TEXT,
        session TEXT,
        emotion TEXT,
        macro_context TEXT,
        outcome TEXT,
        pnl REAL,
        r_multiple REAL,
        execution_quality REAL,
        slippage_bps REAL,
        feature_hash TEXT,
        notes TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_mem_symbol ON institutional_memory(symbol);
    CREATE INDEX IF NOT EXISTS idx_mem_type ON institutional_memory(memory_type);
    CREATE INDEX IF NOT EXISTS idx_mem_pattern ON institutional_memory(pattern);
    CREATE INDEX IF NOT EXISTS idx_mem_regime ON institutional_memory(regime);
    CREATE INDEX IF NOT EXISTS idx_mem_outcome ON institutional_memory(outcome);
    """

    def __init__(self, db_path: str = "data/institutional_memory.db"):
        """Initialize memory database.

        Args:
            db_path: SQLite database path
        """
        self._db_path = db_path
        self._lock = threading.RLock()
        self._cache: List[Memory] = []
        self._init_db()
        self._load_cache()

    def _init_db(self) -> None:
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.executescript(self.SCHEMA)
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("memory_db: init failed: %r", e)

    def _load_cache(self) -> None:
        """Load all memories into RAM for fast search."""
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM institutional_memory ORDER BY timestamp DESC LIMIT 5000"
            ).fetchall()
            for r in rows:
                mem = Memory(
                    id=r["id"], timestamp=r["timestamp"],
                    memory_type=r["memory_type"],
                    symbol=r["symbol"], pattern=r["pattern"],
                    features=json.loads(r["features"] or "{}"),
                    regime=r["regime"], session=r["session"],
                    emotion=r["emotion"], macro_context=r["macro_context"],
                    outcome=r["outcome"], pnl=r["pnl"],
                    r_multiple=r["r_multiple"],
                    execution_quality=r["execution_quality"],
                    slippage_bps=r["slippage_bps"],
                    feature_hash=r["feature_hash"],
                    notes=r["notes"],
                )
                self._cache.append(mem)
            log.info("memory_db: loaded %d memories", len(self._cache))
            conn.close()
        except Exception as e:
            log.warning("memory_db: load failed: %r", e)

    # ------------------------------------------------------------------
    # Store memories
    # ------------------------------------------------------------------
    def store_pattern(self,
                      symbol: str, pattern: str,
                      features: Dict[str, Any],
                      outcome: str, pnl: float, r_multiple: float,
                      regime: str = "", session: str = "",
                      emotion: str = "", macro_context: str = "",
                      execution_quality: float = 0.0,
                      slippage_bps: float = 0.0,
                      notes: str = "") -> str:
        """Store a pattern memory."""
        mem = Memory(
            id=f"mem_{int(time.time()*1000)}_{len(self._cache)}",
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            memory_type="pattern",
            symbol=symbol, pattern=pattern,
            features=features,
            regime=regime, session=session,
            emotion=emotion, macro_context=macro_context,
            outcome=outcome, pnl=pnl, r_multiple=r_multiple,
            execution_quality=execution_quality,
            slippage_bps=slippage_bps,
            feature_hash=self._hash_features(features),
            notes=notes,
        )
        with self._lock:
            self._cache.append(mem)
            if len(self._cache) > 5000:
                self._cache = self._cache[-5000:]
        self._persist(mem)
        return mem.id

    def store_outcome(self, symbol: str, strategy: str,
                      features: Dict[str, Any], outcome: str,
                      pnl: float, r_multiple: float,
                      regime: str = "", session: str = "",
                      execution_quality: float = 0.0) -> str:
        """Store a trade outcome memory."""
        return self.store_pattern(
            symbol=symbol, pattern=strategy,
            features=features, outcome=outcome,
            pnl=pnl, r_multiple=r_multiple,
            regime=regime, session=session,
            execution_quality=execution_quality,
            notes=f"strategy={strategy}",
        )

    def store_execution(self, symbol: str, order_type: str,
                        slippage_bps: float, latency_ms: float,
                        fill_ratio: float, spread_bps: float) -> str:
        """Store an execution memory."""
        return self.store_pattern(
            symbol=symbol, pattern=f"exec_{order_type}",
            features={"slippage": slippage_bps, "latency": latency_ms,
                     "fill_ratio": fill_ratio, "spread": spread_bps},
            outcome="win" if slippage_bps < 3 else "loss",
            pnl=0, r_multiple=0,
            execution_quality=fill_ratio,
            slippage_bps=slippage_bps,
            notes=f"order_type={order_type}, latency={latency_ms}ms",
        )

    # ------------------------------------------------------------------
    # Query memories
    # ------------------------------------------------------------------
    def query_similar(self,
                      symbol: Optional[str] = None,
                      features: Optional[Dict[str, Any]] = None,
                      pattern: Optional[str] = None,
                      regime: Optional[str] = None,
                      top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar memories using feature similarity.

        Uses cosine similarity on feature vectors.
        """
        with self._lock:
            candidates = list(self._cache)

        # Filter by symbol
        if symbol:
            candidates = [m for m in candidates if m.symbol == symbol]
        if pattern:
            candidates = [m for m in candidates if m.pattern == pattern]
        if regime:
            candidates = [m for m in candidates if m.regime == regime]

        if not candidates:
            return []

        # Similarity search
        if features:
            query_vec = self._features_to_vector(features)
            scored = []
            for m in candidates:
                mem_vec = self._features_to_vector(m.features)
                sim = self._cosine_similarity(query_vec, mem_vec)
                scored.append((sim, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            candidates = [m for _, m in scored[:top_k]]
        else:
            candidates = candidates[:top_k]

        return [self._memory_to_dict(m) for m in candidates]

    def query_by_pattern(self, pattern: str,
                         symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for a specific pattern."""
        with self._lock:
            memories = [m for m in self._cache if m.pattern == pattern
                       if symbol is None or m.symbol == symbol]

        if not memories:
            return {"pattern": pattern, "count": 0}

        wins = [m for m in memories if m.outcome == "win"]
        losses = [m for m in memories if m.outcome == "loss"]
        avg_r = np.mean([m.r_multiple for m in memories])
        avg_pnl = np.mean([m.pnl for m in memories])

        return {
            "pattern": pattern,
            "count": len(memories),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / max(len(memories), 1),
            "avg_r": float(avg_r),
            "avg_pnl": float(avg_pnl),
            "avg_execution_quality": float(np.mean([m.execution_quality for m in memories])),
        }

    def query_by_regime(self, regime: str) -> Dict[str, Any]:
        """Get statistics for a specific regime."""
        with self._lock:
            memories = [m for m in self._cache if m.regime == regime]

        if not memories:
            return {"regime": regime, "count": 0}

        wins = [m for m in memories if m.outcome == "win"]
        return {
            "regime": regime,
            "count": len(memories),
            "win_rate": len(wins) / max(len(memories), 1),
            "avg_r": float(np.mean([m.r_multiple for m in memories])),
            "best_pattern": self._best_pattern(memories),
        }

    def _best_pattern(self, memories: List[Memory]) -> Optional[str]:
        """Find the best-performing pattern in a set of memories."""
        by_pattern: Dict[str, list] = defaultdict(list)
        for m in memories:
            by_pattern[m.pattern].append(m.r_multiple)
        if not by_pattern:
            return None
        best = max(by_pattern.items(), key=lambda x: np.mean(x[1]))
        return best[0]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Get memory database statistics."""
        with self._lock:
            total = len(self._cache)
            by_type = defaultdict(int)
            by_symbol = defaultdict(int)
            by_outcome = defaultdict(int)
            for m in self._cache:
                by_type[m.memory_type] += 1
                by_symbol[m.symbol] += 1
                by_outcome[m.outcome] += 1

        return {
            "total_memories": total,
            "by_type": dict(by_type),
            "by_symbol": dict(by_symbol),
            "by_outcome": dict(by_outcome),
            "symbols_in_memory": len(by_symbol),
            "patterns_in_memory": len(set(m.pattern for m in self._cache)),
        }

    def pattern_library(self) -> List[Dict[str, Any]]:
        """Get all patterns with their statistics."""
        with self._lock:
            patterns = set(m.pattern for m in self._cache if m.pattern)
        return [self.query_by_pattern(p) for p in patterns]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _hash_features(self, features: Dict[str, Any]) -> str:
        s = ",".join(f"{k}={features[k]}" for k in sorted(features.keys()))
        return hashlib.md5(s.encode()).hexdigest()[:12]

    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        vals = []
        for k in sorted(features.keys()):
            v = features[k]
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(0.0)
        return np.array(vals, dtype=float)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if len(a) == 0 or len(b) == 0:
            return 0.0
        # Pad shorter vector with zeros
        if len(a) != len(b):
            max_len = max(len(a), len(b))
            a = np.pad(a, (0, max_len - len(a)))
            b = np.pad(b, (0, max_len - len(b)))
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _memory_to_dict(self, m: Memory) -> Dict[str, Any]:
        return {
            "id": m.id, "timestamp": m.timestamp,
            "type": m.memory_type, "symbol": m.symbol,
            "pattern": m.pattern, "features": m.features,
            "regime": m.regime, "session": m.session,
            "outcome": m.outcome, "pnl": m.pnl,
            "r_multiple": m.r_multiple,
            "execution_quality": m.execution_quality,
            "notes": m.notes,
        }

    def _persist(self, mem: Memory) -> None:
        try:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.execute("""
                INSERT OR REPLACE INTO institutional_memory
                (id, timestamp, memory_type, symbol, pattern, features,
                 regime, session, emotion, macro_context, outcome, pnl,
                 r_multiple, execution_quality, slippage_bps, feature_hash, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                mem.id, mem.timestamp, mem.memory_type,
                mem.symbol, mem.pattern,
                json.dumps(mem.features, default=str),
                mem.regime, mem.session, mem.emotion, mem.macro_context,
                mem.outcome, mem.pnl, mem.r_multiple,
                mem.execution_quality, mem.slippage_bps,
                mem.feature_hash, mem.notes,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning("memory_db: persist failed: %r", e)
