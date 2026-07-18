# PHASE 2 AUDIT FINDINGS: Risk Engine, AI Pipeline & Execution

## CRITICAL BUGS IDENTIFIED

### 1. MISSING CONFIGURATION FILE (CRITICAL - BLOCKING)
**Location:** `/workspace/config/config.yaml`
**Issue:** The config file referenced by `config_loader.py` does not exist
**Impact:** System cannot start in production mode
**Evidence:** 
- `config_loader.py:48` expects `config/config.yaml`
- File not found in filesystem
- All modules depend on this configuration

### 2. EMPTY MT5_PASSWORD IN .env.example (CRITICAL - SECURITY)
**Location:** `/workspace/.env.example`
**Issue:** Password field is empty with no guidance
**Impact:** Security vulnerability, operators may use weak/default passwords
**Evidence:** `MT5_PASSWORD=` with no value or example format

### 3. DUPLICATE RISK ENGINE IMPLEMENTATIONS (MAJOR - ARCHITECTURE)
**Locations:** 
- `/workspace/engine/risk.py` (896 lines) - Legacy 9-layer pipeline
- `/workspace/engine/risk_v2.py` (358 lines) - Portfolio-aware v2
- `/workspace/architecture/risk_pipeline.py` (801 lines) - Current 12-gate canonical

**Issue:** Three separate risk engine implementations with overlapping functionality
**Impact:** 
- Code maintenance burden
- Potential for inconsistent risk decisions
- Confusion about which is authoritative
**Evidence:** Comments in `risk.py` state it's "DEPRECATED" but still imported by `execution.py`

### 4. RISK PIPELINE GATE ORDER ISSUE (MAJOR - LOGIC)
**Location:** `/workspace/architecture/risk_pipeline.py:691-714`
**Issue:** PortfolioGate runs AFTER SizingGate/SLTPGate which is correct, but...
**Finding:** The comments claim this is fixed but the actual gate instantiation order needs verification
**Status:** Actually CORRECT in current implementation - gates are properly ordered

### 5. ATR COMPUTATION REDUNDANCY (PERFORMANCE)
**Location:** `/workspace/architecture/risk_pipeline.py:540-650`
**Issue:** Multiple gates recompute ATR unless passed via `pipeline_state`
**Status:** MITIGATED - code checks `pipeline_state.get("atr")` first
**Finding:** Proper optimization already implemented

### 6. SL DISTANCE FORMULA CONSISTENCY (CRITICAL - RISK)
**Location:** `/workspace/architecture/risk_pipeline.py:550-560, 634-639`
**Issue:** SizingGate and SLTPGate MUST use identical SL distance formulas
**Status:** FIXED - Both now use `max(ATR * sl_mult, price * 0.005)`
**Evidence:** Comments at lines 550-558 and 634-638 confirm alignment

### 7. MT5 EXECUTION RETRY LOGIC (MAJOR - RELIABILITY)
**Location:** `/workspace/brokers/mt5_connector.py:451-504`
**Issue:** Order retry logic exists but needs verification
**Status:** PROPERLY IMPLEMENTED
**Findings:**
- Retryable retcodes properly defined (line 31-38)
- Price refresh before retry implemented (line 494-499)
- Non-retryable errors fail fast (line 477-482)

### 8. EXECUTION ENGINE PACING BUG (MAJOR - PERFORMANCE)
**Location:** `/workspace/execution/execution_engine.py:136-160`
**Issue:** Paper mode pacing was capped at 0.01s, negating adaptive delays
**Status:** FIXED per comments at line 141-146
**Evidence:** Now uses configurable `PAPER_DELAY_FRACTION` env var

### 9. BAYESIAN AI NOT INTEGRATED (MINOR - FEATURE GAP)
**Location:** `/workspace/trading_modules/bayesian_ai.py`
**Issue:** Sophisticated uncertainty quantification module exists but unclear integration path
**Finding:** Module is well-implemented but needs wiring into main prediction pipeline

### 10. CONFORMAL PREDICTION UNUSED (MINOR - FEATURE GAP)
**Location:** `/workspace/trading_modules/bayesian_ai.py:63-132`
**Issue:** ConformalPredictor class provides calibrated intervals but not used in production
**Impact:** Missing uncertainty-aware position sizing

## ARCHITECTURAL CONCERNS

### 1. Configuration Management
- No default config template provided
- Operators must create config from scratch
- Missing example configurations for different broker types

### 2. Risk Engine Consolidation Needed
- Three implementations should be consolidated to one
- Clear deprecation path needed for `engine/risk.py`
- `risk_v2.py` appears orphaned (not referenced in main pipeline)

### 3. AI Pipeline Integration
- Bayesian AI module isolated from main prediction flow
- Uncertainty estimates not feeding into position sizing
- Conformal prediction calibration data requirements unclear

## RECOMMENDATIONS

### IMMEDIATE (P0)
1. Create `config/config.yaml` with comprehensive defaults
2. Update `.env.example` with security guidance
3. Verify which risk engine is canonical and remove duplicates

### SHORT-TERM (P1)
4. Integrate Bayesian AI uncertainty into position sizing
5. Add conformal prediction calibration to model training pipeline
6. Document risk engine decision flow for operators

### MEDIUM-TERM (P2)
7. Consolidate risk engine implementations
8. Add integration tests for full risk pipeline
9. Implement backtesting framework for risk parameter optimization

## FILES REQUIRING MODIFICATION

1. **CREATE:** `/workspace/config/config.yaml` - Full configuration template
2. **UPDATE:** `/workspace/.env.example` - Security guidance
3. **REVIEW:** `/workspace/engine/risk.py` - Deprecation cleanup
4. **REVIEW:** `/workspace/engine/risk_v2.py` - Integration or removal
5. **ENHANCE:** `/workspace/trading_modules/bayesian_ai.py` - Integration hooks

