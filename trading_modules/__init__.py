"""
Trading Modules Package v5.0
=============================

36 modules across 7 layers — Autonomous Institutional Crypto AI Platform.

Layers:
    # Execution Layer (11 modules)
    # Validation Layer (5 modules)
    # Intelligence Layer (3 modules)
    # RL Layer (1 module)
    # Advanced AI Layer (7 modules)
    # Crypto Intelligence Layer (1 module) — NEW
    # Self-Management Layer (4 modules) — NEW
"""

# Execution Layer
from .smc_detector import SMCDetector, SMCResult
from .confluence_gate import ConfluenceGate, ConfluenceInput, ConfluenceResult, WeightedConfluenceGate
from .rating_system import RatingSystem, PortfolioRating, RatingResult
from .kill_conditions import KillConditions, PortfolioState, KillDecision
from .bias_tracker import BiasTracker, BiasProfile, DecisionRecord
from .kelly_sizing import kelly_position_size, full_kelly_fraction, r_multiple_position_size, KellyResult
from .r_multiple_tp import RMultipleTP, Position, TPPlan, TPLevel
from .coin_cooldown import CoinCooldownManager, SymbolStats
from .signal_processor import SignalProcessor, TradingDecision
from .cpcv import CPCV
from .trade_pipeline import TradePipeline, PipelineInput, TradeDecision

# Validation Layer
from .edge_thesis import EdgeThesisGate, EdgeThesis, EdgeEvaluation, EdgeCategory
from .platt_calibration import PlattCalibrator, CalibrationResult
from .token_budget import TokenBudget, guarded_call, estimate_cost
from .verified_snapshot import build_verified_snapshot
from .multiple_testing import (
    benjamini_hochberg, deflated_sharpe_ratio,
    whites_reality_check, probability_of_backtest_overfitting,
    verify_backtest, DSRResult, BacktestVerification,
)

# Intelligence Layer
from .triple_barrier import compute_labels, compute_meta_labels, label_distribution, TripleBarrierConfig
from .alpha_zoo import AlphaZoo, rank, scale, ts_rank, ts_corr, delta, decay_linear
from .ml_models import MLModelTrainer, build_features, ModelResult

# RL Layer
from .rl_agent import RLAgent, TradingEnv, PPOConfig, TrainingMetrics

# Advanced AI Layer
from .deep_learning import LSTMForecaster, TransformerForecaster, DeepLearningConfig, ForecastResult
from .bayesian_ai import (
    ConformalPredictor, ConformalInterval, BayesianLinearRegression,
    UncertaintyAwareDecision, make_uncertainty_aware_decision,
)
from .automl import AutoML, ModelTrial
from .genetic_optimizer import GeneticOptimizer, StrategyGenome
from .causal_ai import DoubleML, GrangerCausality, CausalEffect, GrangerResult, causal_summary
from .knowledge_graph import MarketKnowledgeGraph, Entity, Relationship, build_default_market_graph
from .digital_twin import DigitalTwin, TwinPosition, DivergenceTracker

# Crypto Intelligence Layer (v5.0 NEW)
from .crypto_intelligence import (
    PumpDumpDetector, PumpDumpResult,
    RugPullDetector, RugPullResult,
    WhaleTracker, WhaleTransfer,
    StablecoinFlowAnalyzer,
)
from .order_book_ai import OrderBookAnalyzer, OrderBookResult

# Self-Management Layer (v5.0 NEW)
from .self_healing import SelfHealingEngine, Issue, IssueType, IssueSeverity, HealResult
from .continual_learning import ContinualLearner, ModelVersion, UpdateResult
from .drift_detection import DriftDetector, DriftResult
from .model_registry import ModelRegistry, TrainingRun

# Decision Intelligence Layer (v5.0 NEW)
from .counterfactual import CounterfactualAnalyzer, CounterfactualResult
from .swarm_intelligence import StrategySwarm, SwarmDecision, AgentVote

# Integration Bridge
from .integration import (
    enhance_signal, pre_trade_gate, get_smc_context, get_smc_result,
    get_verified_snapshot, calibrate_agent_confidence,
    check_kill_conditions, check_coin_cooldown, record_trade_outcome,
    create_tp_plan, parse_llm_decision,
    get_ml_features, get_triple_barrier_labels, train_ml_models,
    compute_alpha, list_available_alphas,
)

# Paper Trading Engine (v5.1)
from .paper_trading import PaperTradingEngine, PaperPosition, PaperTrade, PortfolioStatus

# Institutional Entry Gate (v5.4) — "Should I trade now?" brain
from .institutional_entry_gate import (
    InstitutionalEntryGate, EntryInput as InstitutionalEntryInput,
    EntryDecision, Action as InstitutionalAction,
)

# v5.5 helper modules
from .market_regime import (
    MarketRegimeDetector, Regime as MarketRegime, regime_allows_trading,
)
from .candle_quality import CandleQualityAnalyzer, CandleQuality
from .correlation_filter import CorrelationFilter, CorrelationResult, build_returns_df
from .trade_cooldown import TradeCooldownManager, CooldownState
from .dynamic_sizing import DynamicPositionSizer, SizingInput, SizingResult

# v5.6 institutional confluence modules
from .volume_profile import VolumeProfileAnalyzer, VolumeProfileResult
from .vwap import VWAPAnalyzer, VWAPResult
from .fibonacci import FibonacciAnalyzer, FibResult
from .wyckoff import WyckoffAnalyzer, WyckoffResult
# Phase 5: fake_breakout archived to legacy/trading_modules_duplicates/
# (kept engine/candlestick/false_breakout.py as canonical)
from .ema_ribbon import EMARibbonAnalyzer, EMARibbonResult, DEFAULT_PERIODS as EMA_DEFAULT_PERIODS
from .cross_asset import CrossAssetConfirmation, CrossAssetResult, DEFAULT_RELATIONSHIPS as CROSS_ASSET_DEFAULTS
from .trade_journal import TradeJournal, TradeRecord, JournalStats

# v5.7 advanced institutional modules
from .auction_market_theory import AMTAnalyzer, AMTResult
from .quant_factors import (
    zscore, zscore_series, hurst_exponent,
    CointegrationResult, cointegration_test,
    PCAResult, pca,
    KalmanResult, kalman_filter,
    HMMResult, hmm_regime,
    BayesianWinRate, bayesian_winrate,
)
from .monte_carlo import MonteCarloSimulator, SimulationInput as MCSimulationInput, SimulationResult as MCSimulationResult
from .explainable_ai import Explainer, Explanation, ExplanationInput
from .ensemble_voting import (
    EnsembleVoter, EnsembleDecision, Vote as EnsembleVote, Model as EnsembleModel,
    trend_model, liquidity_model, risk_model, regime_model,
)
from .liquidation_heatmap import LiquidationHeatmap, LiquidationResult, LiquidationCluster
from .cme_gap import CMEGapDetector, CMEGapResult
from .risk_dashboard import RiskDashboard, PortfolioSnapshot, RiskMetrics

# v5.8 research-grade modules
from .time_series_models import (
    ARIMA, ARIMAResult, GARCH11, GARCHResult,
    StateSpaceModel, StateSpaceResult, holt_winters, HoltWintersResult,
)
from .portfolio_theory import (
    PortfolioResult, markowitz_mpt, min_variance_portfolio,
    black_litterman, risk_parity, hierarchical_risk_parity,
)
from .quant_risk_metrics import (
    calmar_ratio, omega_ratio, ulcer_index, mar_ratio,
    information_ratio, treynor_ratio, sterling_ratio,
    burke_ratio, pain_index, all_metrics,
)
from .execution_algorithms import (
    ExecutionSlice, ExecutionPlan,
    twap_schedule, vwap_schedule, pov_schedule, is_schedule, sniper_schedule,
)
from .statistics_advanced import (
    BootstrapResult, bootstrap_ci, EVTResult, evt_tail_risk,
    CopulaResult, gaussian_copula, GPResult, gaussian_process_fit,
)
from .optimization_meta import (
    OptimizationResult, genetic_algorithm, particle_swarm,
    simulated_annealing, bayesian_optimize,
)
from .nlp_trading import (
    NewsSentimentAnalyzer, NewsAnalysisResult, AggregatedNewsResult,
    BULLISH_WORDS, BEARISH_WORDS, HAWKISH_WORDS, DOVISH_WORDS, EVENT_KEYWORDS,
)
from .chart_patterns import ChartPatternDetector, Pattern, PatternDetectionResult

# v5.9 governance & meta-analysis modules
from .psychology_modeling import PsychologyAnalyzer, PsychologyResult
from .behavioral_finance import BehavioralAnalyzer, TraderHistory, BehavioralResult
from .strategy_decay import StrategyDecayDetector, DecayResult
from .alpha_attribution import AlphaAttribution, AttributionResult
from .tca import TCAAnalyzer, TradeExecution, TCAResult
from .market_calendar import MarketCalendar, CalendarEvent, CalendarResult
from .regime_probability import RegimeProbability, RegimeProbResult
from .failure_analysis import FailureAnalyzer, FailedTrade, FailureResult
from .ai_governance import AIGovernance, GovernanceAction, GovernanceRecord

# v6.0 advanced research modules
from .information_theory import (
    shannon_entropy, kl_divergence, jensen_shannon_divergence,
    mutual_information, transfer_entropy, entropy_rate,
)
from .signal_processing import (
    FFTResult, fft_decompose, haar_wavelet, haar_reconstruct,
    emd_decompose, lowpass_filter, highpass_filter, bandpass_filter,
    hodrick_prescott,
)
from .change_point_detection import ChangePointDetector, ChangePoint
from .anomaly_detection import AnomalyDetector, AnomalyResult
from .game_theory import (
    GameSolution, zero_sum_game_solver, mixed_strategy_nash,
    shapley_value, adversarial_awareness,
)
from .decision_intelligence import (
    expected_utility, optimal_action,
    OptimalStopResult, optimal_stop,
    BanditArm, BanditResult, MultiArmedBandit,
    DecisionNode, evaluate_decision_tree,
    kelly_fraction_multi,
)
from .robust_ai import (
    impute_missing, winsorize, robust_zscore,
    DataQualityReport, DataQualityChecker,
    fail_safe, GracefulDegradation,
    safe_divide, safe_pct_change, clip_to_range,
)
from .synthetic_data import (
    block_bootstrap, simulate_gbm, simulate_jump_diffusion,
    augment_jitter, augment_scaling, augment_time_warp, augment_mixup,
    generate_regime_switching,
)

__version__ = "6.0.0"
__all__ = [
    # Execution Layer
    "SMCDetector", "SMCResult",
    "ConfluenceGate", "ConfluenceInput", "ConfluenceResult", "WeightedConfluenceGate",
    "RatingSystem", "PortfolioRating", "RatingResult",
    "KillConditions", "PortfolioState", "KillDecision",
    "BiasTracker", "BiasProfile", "DecisionRecord",
    "kelly_position_size", "full_kelly_fraction", "r_multiple_position_size", "KellyResult",
    "RMultipleTP", "Position", "TPPlan", "TPLevel",
    "CoinCooldownManager", "SymbolStats",
    "SignalProcessor", "TradingDecision",
    "CPCV", "TradePipeline", "PipelineInput", "TradeDecision",
    # Validation Layer
    "EdgeThesisGate", "EdgeThesis", "EdgeEvaluation", "EdgeCategory",
    "PlattCalibrator", "CalibrationResult",
    "TokenBudget", "guarded_call", "estimate_cost",
    "build_verified_snapshot",
    "benjamini_hochberg", "deflated_sharpe_ratio", "whites_reality_check",
    "probability_of_backtest_overfitting", "verify_backtest",
    "DSRResult", "BacktestVerification",
    # Intelligence Layer
    "compute_labels", "compute_meta_labels", "label_distribution", "TripleBarrierConfig",
    "AlphaZoo", "rank", "scale", "ts_rank", "ts_corr", "delta", "decay_linear",
    "MLModelTrainer", "build_features", "ModelResult",
    # RL Layer
    "RLAgent", "TradingEnv", "PPOConfig", "TrainingMetrics",
    # Advanced AI Layer
    "LSTMForecaster", "TransformerForecaster", "DeepLearningConfig", "ForecastResult",
    "ConformalPredictor", "ConformalInterval", "BayesianLinearRegression",
    "UncertaintyAwareDecision", "make_uncertainty_aware_decision",
    "AutoML", "ModelTrial",
    "GeneticOptimizer", "StrategyGenome",
    "DoubleML", "GrangerCausality", "CausalEffect", "GrangerResult", "causal_summary",
    "MarketKnowledgeGraph", "Entity", "Relationship", "build_default_market_graph",
    "DigitalTwin", "TwinPosition", "DivergenceTracker",
    # Crypto Intelligence Layer (v5.0)
    "PumpDumpDetector", "PumpDumpResult",
    "RugPullDetector", "RugPullResult",
    "WhaleTracker", "WhaleTransfer",
    "StablecoinFlowAnalyzer",
    "OrderBookAnalyzer", "OrderBookResult",
    # Self-Management Layer (v5.0)
    "SelfHealingEngine", "Issue", "IssueType", "IssueSeverity", "HealResult",
    "ContinualLearner", "ModelVersion", "UpdateResult",
    "DriftDetector", "DriftResult",
    "ModelRegistry", "TrainingRun",
    # Decision Intelligence Layer (v5.0)
    "CounterfactualAnalyzer", "CounterfactualResult",
    "StrategySwarm", "SwarmDecision", "AgentVote",
    # Integration
    "enhance_signal", "pre_trade_gate", "get_smc_context", "get_smc_result",
    "get_verified_snapshot", "calibrate_agent_confidence",
    "check_kill_conditions", "check_coin_cooldown", "record_trade_outcome",
    "create_tp_plan", "parse_llm_decision",
    "get_ml_features", "get_triple_barrier_labels", "train_ml_models",
    "compute_alpha", "list_available_alphas",
    # Paper Trading Engine (v5.1)
    "PaperTradingEngine", "PaperPosition", "PaperTrade", "PortfolioStatus",
    # Institutional Entry Gate (v5.4)
    "InstitutionalEntryGate", "InstitutionalEntryInput", "EntryDecision", "InstitutionalAction",
    # v5.5 helpers
    "MarketRegimeDetector", "MarketRegime", "regime_allows_trading",
    "CandleQualityAnalyzer", "CandleQuality",
    "CorrelationFilter", "CorrelationResult", "build_returns_df",
    "TradeCooldownManager", "CooldownState",
    "DynamicPositionSizer", "SizingInput", "SizingResult",
    # v5.6 institutional confluence
    "VolumeProfileAnalyzer", "VolumeProfileResult",
    "VWAPAnalyzer", "VWAPResult",
    "FibonacciAnalyzer", "FibResult",
    "WyckoffAnalyzer", "WyckoffResult",
    "FakeBreakoutDetector", "BreakoutResult",
    "EMARibbonAnalyzer", "EMARibbonResult", "EMA_DEFAULT_PERIODS",
    "CrossAssetConfirmation", "CrossAssetResult", "CROSS_ASSET_DEFAULTS",
    "TradeJournal", "TradeRecord", "JournalStats",
    # v5.7 advanced institutional
    "AMTAnalyzer", "AMTResult",
    "zscore", "zscore_series", "hurst_exponent",
    "CointegrationResult", "cointegration_test",
    "PCAResult", "pca",
    "KalmanResult", "kalman_filter",
    "HMMResult", "hmm_regime",
    "BayesianWinRate", "bayesian_winrate",
    "MonteCarloSimulator", "MCSimulationInput", "MCSimulationResult",
    "Explainer", "Explanation", "ExplanationInput",
    "EnsembleVoter", "EnsembleDecision", "EnsembleVote", "EnsembleModel",
    "trend_model", "liquidity_model", "risk_model", "regime_model",
    "LiquidationHeatmap", "LiquidationResult", "LiquidationCluster",
    "CMEGapDetector", "CMEGapResult",
    "RiskDashboard", "PortfolioSnapshot", "RiskMetrics",
    # v5.8 research-grade
    "ARIMA", "ARIMAResult", "GARCH11", "GARCHResult",
    "StateSpaceModel", "StateSpaceResult", "holt_winters", "HoltWintersResult",
    "PortfolioResult", "markowitz_mpt", "min_variance_portfolio",
    "black_litterman", "risk_parity", "hierarchical_risk_parity",
    "calmar_ratio", "omega_ratio", "ulcer_index", "mar_ratio",
    "information_ratio", "treynor_ratio", "sterling_ratio",
    "burke_ratio", "pain_index", "all_metrics",
    "ExecutionSlice", "ExecutionPlan",
    "twap_schedule", "vwap_schedule", "pov_schedule", "is_schedule", "sniper_schedule",
    "BootstrapResult", "bootstrap_ci", "EVTResult", "evt_tail_risk",
    "CopulaResult", "gaussian_copula", "GPResult", "gaussian_process_fit",
    "OptimizationResult", "genetic_algorithm", "particle_swarm",
    "simulated_annealing", "bayesian_optimize",
    "NewsSentimentAnalyzer", "NewsAnalysisResult", "AggregatedNewsResult",
    "BULLISH_WORDS", "BEARISH_WORDS", "HAWKISH_WORDS", "DOVISH_WORDS", "EVENT_KEYWORDS",
    "ChartPatternDetector", "Pattern", "PatternDetectionResult",
    # v5.9 governance & meta-analysis
    "PsychologyAnalyzer", "PsychologyResult",
    "BehavioralAnalyzer", "TraderHistory", "BehavioralResult",
    "StrategyDecayDetector", "DecayResult",
    "AlphaAttribution", "AttributionResult",
    "TCAAnalyzer", "TradeExecution", "TCAResult",
    "MarketCalendar", "CalendarEvent", "CalendarResult",
    "RegimeProbability", "RegimeProbResult",
    "FailureAnalyzer", "FailedTrade", "FailureResult",
    "AIGovernance", "GovernanceAction", "GovernanceRecord",
    # v6.0 advanced research
    "shannon_entropy", "kl_divergence", "jensen_shannon_divergence",
    "mutual_information", "transfer_entropy", "entropy_rate",
    "FFTResult", "fft_decompose", "haar_wavelet", "haar_reconstruct",
    "emd_decompose", "lowpass_filter", "highpass_filter", "bandpass_filter",
    "hodrick_prescott",
    "ChangePointDetector", "ChangePoint",
    "AnomalyDetector", "AnomalyResult",
    "GameSolution", "zero_sum_game_solver", "mixed_strategy_nash",
    "shapley_value", "adversarial_awareness",
    "expected_utility", "optimal_action",
    "OptimalStopResult", "optimal_stop",
    "BanditArm", "BanditResult", "MultiArmedBandit",
    "DecisionNode", "evaluate_decision_tree", "kelly_fraction_multi",
    "impute_missing", "winsorize", "robust_zscore",
    "DataQualityReport", "DataQualityChecker",
    "fail_safe", "GracefulDegradation",
    "safe_divide", "safe_pct_change", "clip_to_range",
    "block_bootstrap", "simulate_gbm", "simulate_jump_diffusion",
    "augment_jitter", "augment_scaling", "augment_time_warp", "augment_mixup",
    "generate_regime_switching",
]
