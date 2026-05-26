# Daydream Handler – Prompt Engineering, Epochs & Episode Planning

## Overview

This document defines the prompt engineering sequence for training the
Layer0 and Grand Unified models, including epoch structure, episode
design, TODO lists, and scheduled actions.

---

## Model Hierarchy

```
msomatothing/layer0 (HuggingFace)
├── layer0_merged          ← Primary inference model
│   ├── Chronos            ← Time-series forecasting backbone
│   ├── TimeSeries         ← LSTM/Transformer sequence encoder
│   ├── DeepHermes         ← Instruction-following reasoning layer
│   ├── MTF                ← Multi-timeframe fusion head
│   ├── NumpyLogistics     ← Logistic regression baseline
│   ├── Regressors         ← Ensemble regression heads
│   └── GradientBooster    ← XGBoost/LightGBM meta-learner
└── grand_unified          ← Full system model
    ├── All layer0 sub-systems
    ├── GNN                ← Graph Neural Network (market topology)
    └── RL Readout         ← Reinforcement learning policy head
```

---

## Training Phase Plan

### Phase 0 – Replay (LFO Jitter) – **Current**

**Goal:** Establish baseline price-action pattern recognition.

**Prompt Template (per bar):**
```json
{
  "context": "OHLCV window [50 bars], MTF H1/H4/D1",
  "indicators": ["ichimoku", "fib_levels", "vol_cloud", "sd_zones"],
  "task": "classify_direction",
  "classes": ["HOLD", "BUY", "SELL"],
  "target_accuracy": 0.99,
  "target_precision": 0.95
}
```

**Epoch Structure:**
| Epoch | Bars/Symbol | LFO Amplitude | Noise Sigma | Focus |
|-------|-------------|---------------|-------------|-------|
| 1     | 5000        | 0.0002        | 0.0001      | Pattern warmup |
| 2     | 10000       | 0.0004        | 0.0002      | Trend detection |
| 3     | 20000       | 0.0006        | 0.0003      | S/D zone learning |
| 4     | 50000       | 0.0008        | 0.0004      | Multi-symbol fusion |
| 5     | Full hist.  | 0.001         | 0.0005      | Full system eval |

---

### Phase 1 – Active RL Mode (Planned)

**Goal:** Online reward-driven policy improvement.

**Reward Signal:**
```
R = Σ [
  +1.0 × (TP hit) ×  RR_ratio,
  -1.0 × (SL hit),
  -0.1 × (spread cost),
  +0.5 × (correct direction within N bars),
  -0.5 × (missed entry after strong signal)
]
```

**Episode Structure:**
| Episode | Duration | Mode | Checkpoints |
|---------|----------|------|-------------|
| E1      | 500 bars | Exploration (ε=0.9) | Every 100 bars |
| E2      | 1000 bars | Mixed (ε=0.5) | Every 200 bars |
| E3      | 2000 bars | Exploitation (ε=0.1) | Every 500 bars |
| E∞      | Live stream | Full exploit | Every trade close |

---

## TODO – Development Roadmap

### Core Engine
- [x] TimeSeries + LFO Jitter Replay
- [x] Ichimoku, Fibonacci, Volatility Cloud
- [x] Supply/Demand zone detector
- [x] Liquidity sweep detector
- [x] Stub model inference
- [x] Feature composer (GEMM/MatMul/ReLU/Concat)
- [x] Strategy multiplexer + DT gate
- [x] Risk manager (polygonal SL/TP)
- [x] Redis client + checkpoint manager
- [x] Hyper-extensive telemetry
- [x] LibTorch model loading (`TorchModel::load`)
- [x] SafeTensors weight file reader (`SafeTensorsFile::parse_header`, `read_tensor_f32`)
- [x] Native model loader (`ModelLoader::discover_and_load`)
- [x] ONNX Runtime native inference backend (`OnnxModel`, `USE_ONNXRUNTIME`)
- [x] Training log CSV / telemetry JSONL import into Telemetry system
- [ ] HuggingFace model downloader (online fetch from msomatothing/layer0)
- [ ] Broker plugin interface (live mode)
- [ ] WebSocket market data feed
- [ ] RL environment wrapper

### ML / Training
- [x] Safetensors weight file reader
- [x] Native model loader (SafeTensorsModel, OnnxModel)
- [x] Training log / telemetry JSONL import
- [ ] Custom Chronos adaptor for OHLCV input
- [ ] GNN market topology graph builder
- [ ] Gradient booster meta-learner wrapper
- [ ] Online learning (continual fine-tuning)
- [ ] Model versioning + A/B testing

### Infrastructure
- [ ] Redis Streams for real-time bar ingestion
- [ ] Prometheus metrics exporter endpoint
- [ ] TensorBoard log converter (JSONL → tfevents)
- [ ] Docker Compose (engine + Redis + monitoring)
- [ ] Kubernetes deployment manifest

### Testing
- [x] Unit tests: timeseries, indicators, risk, inference, checkpoint
- [ ] Integration tests: full replay cycle
- [ ] Property-based tests (fuzz price series)
- [ ] Benchmark: bars/second throughput

---

## Scheduled Actions

```yaml
# Defined in .github/workflows/debug.yml and main.yml
nightly_02_00:  cmake build + full unit test suite
nightly_06_00:  health check + memory-check (valgrind)
weekly_sunday:  full historical replay across all 9 symbols
on_push_main:   build → test → static analysis → release artifact
on_pr:          build → test → cppcheck → review comment
```

---

## HParam Sweep Configuration

For future optuna/Ray Tune integration:

```json
{
  "search_space": {
    "risk_pct":            [0.005, 0.01, 0.02],
    "feature_window":      [20, 50, 100],
    "project_dim":         [32, 64, 128],
    "tenkan_period":       [7, 9, 12],
    "kijun_period":        [20, 26, 30],
    "breakout_threshold":  [0.0005, 0.001, 0.002],
    "lfo_amplitude":       [0.0001, 0.0005, 0.001],
    "min_confidence":      [0.50, 0.55, 0.60, 0.65]
  },
  "metric":     "trade/closed_pnl",
  "direction":  "maximize",
  "n_trials":   100
}
```

---

## Continuum Learning Strategy

The system implements continuum learning through:

1. **Checkpoint rotation** – latest N weight snapshots retained
2. **Online telemetry** – every metric logged for post-hoc analysis
3. **Interaction time tracking** – timing events stored per component
4. **hparam dynamicity** – hparams logged at each checkpoint for drift detection
5. **Replay augmentation** – LFO jitter prevents catastrophic forgetting
6. **Safetensors snapshots** – binary-compatible with Python training loop

```
Learning Loop:
  replay → evaluate → checkpoint → adjust hparams → replay
       ↑                                                  │
       └──────────────────── RL reward signal ───────────┘
```
