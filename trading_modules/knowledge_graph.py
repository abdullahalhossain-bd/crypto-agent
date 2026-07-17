"""
Knowledge Graph Module — Market Entity Relationships
=====================================================

Builds a graph of market entities and their relationships:
  Fed → USD → Gold → Bond → Oil → JPY

Nodes: Assets, indicators, events, institutions
Edges: Causal links, correlations, leading/lagging relationships

Source: ml4t-3e (review #18) ch.23 — Knowledge Graphs
        Vibe-Trading (review #23) — GraphRAG concept

Usage:
    from trading_modules.knowledge_graph import MarketKnowledgeGraph

    kg = MarketKnowledgeGraph()

    # Add entities
    kg.add_entity("BTC", type="asset")
    kg.add_entity("Funding Rate", type="indicator")
    kg.add_entity("USD", type="currency")
    kg.add_entity("Fed", type="institution")

    # Add relationships
    kg.add_edge("Fed", "USD", relation="controls", strength=0.9)
    kg.add_edge("USD", "Gold", relation="inversely_correlated", strength=-0.7)
    kg.add_edge("Funding Rate", "BTC", relation="affects", strength=0.5)
    kg.add_edge("BTC", "ETH", relation="correlated", strength=0.85)

    # Query: what affects BTC?
    upstream = kg.get_upstream("BTC")
    # → ["Funding Rate", "USD", "Fed"]

    # Get relationship chain
    chain = kg.get_path("Fed", "BTC")
    # → ["Fed", "USD", "Gold", "BTC"]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A market entity (asset, indicator, institution, event)."""
    name: str
    type: str = "asset"  # asset, indicator, currency, institution, event, macro
    attributes: dict = field(default_factory=dict)


@dataclass
class Relationship:
    """A directed relationship between two entities."""
    source: str
    target: str
    relation: str  # "affects", "correlated", "inversely_correlated", "causes", "leads"
    strength: float = 0.0  # -1 to 1 (negative = inverse)
    lag: int = 0  # Time lag in bars/days
    description: str = ""


class MarketKnowledgeGraph:
    """
    Market Knowledge Graph for storing and querying entity relationships.

    Supports:
      - Add entities and relationships
      - Query upstream (what affects X?)
      - Query downstream (what does X affect?)
      - Find paths between entities
      - Get correlation chains
      - Visualize as text
    """

    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.edges: list[Relationship] = []
        self._adjacency: dict[str, list[Relationship]] = defaultdict(list)
        self._reverse_adjacency: dict[str, list[Relationship]] = defaultdict(list)

    def add_entity(self, name: str, type: str = "asset", **attributes) -> None:
        """Add an entity to the graph."""
        self.entities[name] = Entity(name=name, type=type, attributes=attributes)

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str = "affects",
        strength: float = 0.0,
        lag: int = 0,
        description: str = "",
    ) -> None:
        """Add a directed relationship."""
        # Ensure both entities exist
        if source not in self.entities:
            self.add_entity(source)
        if target not in self.entities:
            self.add_entity(target)

        rel = Relationship(
            source=source, target=target, relation=relation,
            strength=strength, lag=lag, description=description,
        )
        self.edges.append(rel)
        self._adjacency[source].append(rel)
        self._reverse_adjacency[target].append(rel)

    def get_upstream(self, entity: str, depth: int = 3) -> list[str]:
        """Get all entities that affect the given entity (upstream)."""
        visited = set()
        queue = deque([(entity, 0)])

        while queue:
            current, d = queue.popleft()
            if d >= depth:
                continue
            for rel in self._reverse_adjacency.get(current, []):
                if rel.source not in visited:
                    visited.add(rel.source)
                    queue.append((rel.source, d + 1))

        return list(visited)

    def get_downstream(self, entity: str, depth: int = 3) -> list[str]:
        """Get all entities affected by the given entity (downstream)."""
        visited = set()
        queue = deque([(entity, 0)])

        while queue:
            current, d = queue.popleft()
            if d >= depth:
                continue
            for rel in self._adjacency.get(current, []):
                if rel.target not in visited:
                    visited.add(rel.target)
                    queue.append((rel.target, d + 1))

        return list(visited)

    def get_path(self, source: str, target: str) -> Optional[list[str]]:
        """Find shortest path from source to target."""
        if source == target:
            return [source]

        visited = {source}
        queue = deque([(source, [source])])

        while queue:
            current, path = queue.popleft()
            for rel in self._adjacency.get(current, []):
                if rel.target == target:
                    return path + [rel.target]
                if rel.target not in visited:
                    visited.add(rel.target)
                    queue.append((rel.target, path + [rel.target]))

        return None

    def get_relationships(self, entity: str) -> list[Relationship]:
        """Get all direct relationships for an entity."""
        return self._adjacency.get(entity, []) + self._reverse_adjacency.get(entity, [])

    def get_correlation_chain(self, entity1: str, entity2: str) -> Optional[list[str]]:
        """Find correlation chain between two entities."""
        return self.get_path(entity1, entity2)

    def get_impact_analysis(self, entity: str) -> dict:
        """
        Analyze the full impact of an entity on the market.

        Returns upstream (what affects it) and downstream (what it affects).
        """
        return {
            "entity": entity,
            "type": self.entities.get(entity, Entity(name=entity)).type,
            "upstream": self.get_upstream(entity),
            "downstream": self.get_downstream(entity),
            "direct_relationships": [
                {
                    "direction": "→" if r.source == entity else "←",
                    "other": r.target if r.source == entity else r.source,
                    "relation": r.relation,
                    "strength": r.strength,
                }
                for r in self.get_relationships(entity)
            ],
        }

    def get_context_for_prompt(self, entity: str) -> str:
        """Generate knowledge graph context for LLM prompts."""
        impact = self.get_impact_analysis(entity)
        lines = [f"## Knowledge Graph Context for {entity}"]

        if impact["upstream"]:
            lines.append(f"**Upstream (what affects {entity})**: {', '.join(impact['upstream'])}")

        if impact["downstream"]:
            lines.append(f"**Downstream (what {entity} affects)**: {', '.join(impact['downstream'])}")

        if impact["direct_relationships"]:
            lines.append("\n**Direct Relationships**:")
            for rel in impact["direct_relationships"]:
                arrow = "→" if rel["direction"] == "→" else "←"
                strength_str = f" (strength: {rel['strength']:.1f})" if rel["strength"] != 0 else ""
                lines.append(f"  {entity} {arrow} {rel['other']} [{rel['relation']}]{strength_str}")

        return "\n".join(lines)

    def summary(self) -> dict:
        """Get graph summary statistics."""
        return {
            "n_entities": len(self.entities),
            "n_relationships": len(self.edges),
            "entity_types": list(set(e.type for e in self.entities.values())),
            "relation_types": list(set(r.relation for r in self.edges)),
        }


def build_default_market_graph() -> MarketKnowledgeGraph:
    """
    Build a default market knowledge graph with common relationships.

    This provides a starting point — extend with domain-specific knowledge.
    """
    kg = MarketKnowledgeGraph()

    # Entities
    kg.add_entity("Fed", type="institution")
    kg.add_entity("ECB", type="institution")
    kg.add_entity("USD", type="currency")
    kg.add_entity("EUR", type="currency")
    kg.add_entity("JPY", type="currency")
    kg.add_entity("BTC", type="asset")
    kg.add_entity("ETH", type="asset")
    kg.add_entity("Gold", type="asset")
    kg.add_entity("Oil", type="asset")
    kg.add_entity("S&P 500", type="index")
    kg.add_entity("Nasdaq", type="index")
    kg.add_entity("VIX", type="indicator")
    kg.add_entity("US 10Y", type="indicator")
    kg.add_entity("DXY", type="indicator")
    kg.add_entity("Funding Rate", type="indicator")
    kg.add_entity("NFP", type="event")
    kg.add_entity("CPI", type="event")

    # Relationships
    kg.add_edge("Fed", "USD", "controls", 0.9, description="Fed policy drives USD strength")
    kg.add_edge("Fed", "US 10Y", "sets", 0.95, description="Fed sets interest rates")
    kg.add_edge("Fed", "S&P 500", "affects", 0.7, description="Rate decisions affect equities")
    kg.add_edge("USD", "Gold", "inversely_correlated", -0.7, description="Strong USD = weak Gold")
    kg.add_edge("USD", "BTC", "inversely_correlated", -0.5, description="Strong USD pressures BTC")
    kg.add_edge("USD", "Oil", "inversely_correlated", -0.6, description="Oil priced in USD")
    kg.add_edge("DXY", "USD", "measures", 0.95, description="DXY is USD index")
    kg.add_edge("US 10Y", "Gold", "inversely_correlated", -0.4, description="Higher yields = lower Gold")
    kg.add_edge("VIX", "S&P 500", "inversely_correlated", -0.8, description="Fear index vs stocks")
    kg.add_edge("VIX", "BTC", "inversely_correlated", -0.3, description="Risk-off hurts BTC")
    kg.add_edge("BTC", "ETH", "correlated", 0.85, description="High crypto correlation")
    kg.add_edge("Funding Rate", "BTC", "affects", 0.4, description="High funding = leveraged longs")
    kg.add_edge("NFP", "USD", "affects", 0.6, description="Strong jobs = strong USD")
    kg.add_edge("NFP", "Gold", "affects", -0.4, description="Strong jobs = weak Gold")
    kg.add_edge("CPI", "Fed", "affects", 0.7, description="High CPI = Fed hawkish")
    kg.add_edge("CPI", "Gold", "affects", 0.3, description="Inflation hedge")
    kg.add_edge("Oil", "CPI", "affects", 0.5, description="Oil drives inflation")
    kg.add_edge("ECB", "EUR", "controls", 0.9, description="ECB policy drives EUR")
    kg.add_edge("S&P 500", "Nasdaq", "correlated", 0.9, description="Tech-heavy correlation")
    kg.add_edge("S&P 500", "BTC", "correlated", 0.4, description="Risk-on correlation")

    return kg
