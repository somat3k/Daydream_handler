# Daydream Handler – Architecture & Design Documentation

## Overview

Daydream Handler is an institutional-grade C++ algorithmic trading system with:
- ML inference pipeline (HuggingFace `msomatothing/layer0`)
- Multi-timeframe strategy multiplexer
- Real-time and replay execution modes
- Redis-backed state management
- Hyper-extensive telemetry and checkpointing
- GitHub Actions as primary execution runtime

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       CLI / Main Entry                          │
│                    (src/cli/cli.hpp)                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                     Trading Engine                              │
│                (src/core/engine/)                               │
│  • Orchestrates all components                                  │
│  • Manages positions, P&L, checkpoints                         │
│  • Parallel symbol processing                                   │
└─────┬──────────────┬──────────────────┬──────────────────┬──────┘
      │              │                  │                  │
┌─────▼─────┐  ┌─────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
│  Data     │  │ Indicators │  │  Strategy    │  │  ML Inference│
│ Pipeline  │  │ Module     │  │  Multiplexer │  │  Engine      │
│(data/)    │  │(indicators)│  │ (strategy/)  │  │ (ml/)        │
│• OHLCV    │  │• Ichimoku  │  │• Breakout    │  │• Layer0      │
│• MTF      │  │• Fib       │  │• Reversal    │  │• GrandUnified│
│• LFO Jitr │  │• VolCloud  │  │• S/D Zone    │  │• FeatureComp │
│• Timesers │  │• S&D       │  │• DT Gate     │  │• GEMM/MatMul │
└─────┬─────┘  └─────┬──────┘  └───────┬──────┘  └───────┬──────┘
      │              │                  │                  │
┌─────▼──────────────▼──────────────────▼──────────────────▼──────┐
│                       Risk Manager                              │
│              (src/core/risk/risk_manager.hpp)                   │
│  • Position sizing (polygonal structure matching)              │
│  • SL behind last swing peak/deep                              │
│  • TP: 1–80 pip institutional range                            │
│  • Drawdown limits, spread filter                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    Storage Layer                                │
│              (src/storage/)                                     │
│  • Redis Client (hiredis + in-memory fallback)                 │
│  • Checkpoint Manager (safetensors-compatible JSON)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                 Logging & Telemetry                             │
│              (src/logging/)                                     │
│  • spdlog (console + rotating file)                            │
│  • Telemetry: scalars, histograms, hparams, timing events      │
│  • TensorBoard-compatible JSONL output                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Data Pipeline (`src/data/`)

| File | Purpose |
|------|---------|
| `market_data.hpp` | OHLCV Bar, Tick, Instrument, Timeframe definitions |
| `timeseries.hpp` | Sliding-window `TimeSeries`; LFO Jitter Replay engine |
| `data_pipeline.hpp` | Feed manager – CSV/JSON/Redis/Broker sources |

**Instruments:** XAUUSD, XAUEUR, XAGUSD, XAGEUR, GBPUSD, EURUSD, BTCUSD, ETHUSD, HYPEUSD  
**Timeframes:** M1, M5, M15, M30, H1, H4, D1, W1, MN

**LFO Jitter Replay:** Early-stage augmentation injects sinusoidal LFO noise plus Gaussian perturbation into replay bars to simulate market microstructure noise during initial training.

---

### 2. Indicators (`src/core/indicators/`)

| Indicator | File | Description |
|-----------|------|-------------|
| Ichimoku Cloud | `ichimoku.hpp` | Tenkan/Kijun/Senkou A+B/Chikou; cloud bias |
| Fibonacci | `fibonacci.hpp` | Retracements + extensions (7 standard levels) |
| Volatility Cloud | `volatility_cloud.hpp` | ATR-based adaptive envelope; expanding/contracting |
| Supply & Demand | `supply_demand.hpp` | Impulse + consolidation zone detection |
| Liquidity Sweep | `liquidity.hpp` | Equal H/L, stop hunts, spread spikes |

---

### 3. ML Inference (`src/ml/`)

#### Models (HuggingFace `msomatothing/layer0`)

| Model | Key | Description |
|-------|-----|-------------|
| Layer0 Merged | `layer0_merged` | Merged ensemble: Chronos + TimeSeries + DeepHermes + MTF + Numpy Logistics + Regressors + Gradient Booster |
| Grand Unified | `grand_unified` | Full unified model: all Layer0 sub-systems + GNN + RL readout |

#### Feature Composer
- Builds feature tensors from OHLCV windows + indicator values
- **GEMM** (General Matrix Multiply) projection via `gemm()`
- **MatMul** via shape_to() + GEMM pipeline
- **ReLU** activation
- **Concat** (feature vector concatenation)
- **Shapers** (`shape_to()`) and **Splitters** (`split_at()`)
- Softmax for probability output

#### Backends
- **LibTorch** (primary): TorchScript `.pt` models  
- **Stub** (fallback/CI): heuristic mean-trend classifier

---

### 4. Strategy System (`src/core/strategy/`)

Three base strategies feed a **StrategyMultiplexer**:

```
Breakout Strategy ─────┐
Reversal Strategy ──────┤──► StrategyMultiplexer ──► Risk Manager ──► Position
S/D Zone Strategy ─────┘           │
                               ML Gate (Layer0)
                               DT Gate (pre-filter)
                               Min confidence 55%
```

- **Breakout:** Range breakout with `m_p.threshold` confirmation  
- **Reversal:** Ichimoku cloud + volatility contraction  
- **S/D Zone:** Supply/Demand zone touch + strength scoring  
- **Signal confidence** blended 60% strategy / 40% ML model

---

### 5. Risk Manager (`src/core/risk/`)

- `risk_pct = 1%` default account risk per trade
- SL placed **1 pip behind last swing peak/deep**
- TP clamped to institutional range: **1–80 pips** (configurable)
- Lot size = `risk_amount / (sl_pips × pip_value)`
- Spread filter: reject if spread > `max_spread_pips`
- Drawdown guard: reject if DD > `max_dd_pct`

---

### 6. Storage (`src/storage/`)

#### Redis Client
- Full `SET/GET/DEL/EXISTS/LPUSH/RPUSH/LRANGE/INCR/PUBLISH` API
- In-memory fallback when `USE_REDIS=OFF`
- Used for: bar caching, state persistence, pub/sub notifications

#### Checkpoint Manager
- Saves JSON snapshots every N bars (default: 1000)
- **Safetensors-compatible convention**: `__version__`, `__type__` header
- Rotation: keeps last N checkpoints (default: 5)
- Includes: step, P&L, open positions, telemetry snapshot, hparams

---

### 7. Logging & Telemetry (`src/logging/`)

#### Logger (spdlog)
- Console sink (coloured, configurable level)
- Rotating file sink (50 MB × 10 files)
- Named child loggers per component

#### Telemetry (Hyper-Extensive Mode)
Emits structured JSONL files compatible with TensorBoard / Prometheus:

| Output File | Content |
|-------------|---------|
| `logs/telemetry/scalars.jsonl` | All scalar metrics (tag, value, step, wall_ms) |
| `logs/telemetry/hparams.json` | All hyperparameters |
| `logs/telemetry/events.jsonl` | Timing events (elapsed_us per operation) |

Tracked metrics:
- `engine/total_pnl`, `engine/open_positions`
- `trade/entry_price`, `trade/lot_size`, `trade/rr_ratio`, `trade/closed_pnl`
- `strategy/confidence`, `strategy/direction`
- `epoch/pnl`
- Histogram summaries: mean, std, min, max per tag

---

## Build System

### CMake (Primary)

```bash
# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=ON -DUSE_REDIS=OFF -DUSE_LIBTORCH=OFF

# Build
cmake --build build --parallel

# Test
cd build && ctest --output-on-failure

# Install
cmake --install build --prefix /usr/local
```

### Bazel (Secondary)

```bash
bazel build //:daydream_handler
bazel test //tests/unit:all_unit_tests
```

---

## GitHub Actions Workflows

| Workflow | File | Trigger |
|----------|------|---------|
| CI – Build, Test & Deploy | `main.yml` | push/PR/schedule |
| Debug – Auto-Scan & Copilot PR | `debug.yml` | workflow_run/schedule |
| ML – Training & Checkpoints | `ml_train.yml` | push to ml/, manual |

---

## Secrets (GitHub Actions)

| Secret | Usage |
|--------|-------|
| `BROKER_API_KEY` | Live broker connection (future) |
| `BROKER_API_SECRET` | Live broker authentication (future) |
| `REDIS_PASSWORD` | Production Redis authentication |
| `HF_TOKEN` | HuggingFace model download token |

Configure at: `Settings → Secrets and Variables → Actions`

---

## Reinforcement Learning Integration (Planned)

The system is architected for RL extension:
- `write_on` / `read_on` state management via checkpoint system
- Safetensors-compatible weight snapshots
- Replay buffer via `LFOJitterReplay`
- Telemetry hparams for reward tracking
- Active mode: replace stub model with RL policy network via LibTorch
