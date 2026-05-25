# Daydream Handler

Institutional-grade C++ algorithmic trading system with ML inference, multi-timeframe strategy execution, and GitHub Actions as primary runtime.

[![CI](https://github.com/somat3k/Daydream_handler/actions/workflows/main.yml/badge.svg)](https://github.com/somat3k/Daydream_handler/actions/workflows/main.yml)

---

## Quick Start

```bash
# Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TESTS=ON -DUSE_REDIS=OFF -DUSE_LIBTORCH=OFF
cmake --build build --parallel

# Test
cd build && ctest --output-on-failure

# Run
./build/src/daydream_handler help
./build/src/daydream_handler info
./build/src/daydream_handler replay --symbol XAUUSD --epochs 1
```

## Features

- **9 Instruments:** XAUUSD, XAUEUR, XAGUSD, XAGEUR, GBPUSD, EURUSD, BTCUSD, ETHUSD, HYPEUSD
- **All timeframes:** M1 → MN
- **ML Models:** Layer0 Merged + Grand Unified (HuggingFace `msomatothing/layer0`)
  - Chronos, TimeSeries, DeepHermes, MTF, NumpyLogistics, Regressors, GradientBooster
- **Indicators:** Ichimoku Cloud, Fibonacci, Volatility Cloud, Supply/Demand, Liquidity Sweep
- **Feature Engineering:** GEMM, MatMul, ReLU, Concat, Shapers, Splitters
- **Strategy System:** Breakout, Reversal, S/D Zone + Decision Tree gate + ML gate
- **Risk Management:** Polygonal structure SL/TP (1–80 pip institutional range)
- **Storage:** Redis + in-memory fallback + safetensors-compatible checkpoints
- **Telemetry:** Hyper-extensive JSONL (scalars, histograms, hparams, timing events)
- **LFO Jitter Replay:** Early-stage augmented timeseries training

## Documentation

| Document | Path |
|----------|------|
| Architecture | `docs/architecture.md` |
| Settings Reference | `docs/settings.md` |
| Prompt Engineering & Epochs | `docs/prompt_engineering.md` |

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `USE_LIBTORCH` | ON | Enable LibTorch ML backend |
| `USE_REDIS` | ON | Enable hiredis Redis client |
| `ENABLE_TESTS` | ON | Build unit tests |
| `ENABLE_TELEMETRY` | ON | Enable telemetry output |

## GitHub Actions

| Workflow | Description |
|----------|-------------|
| `main.yml` | CI: Build, test, static analysis, release |
| `debug.yml` | Auto-scan logs + Copilot debug PR creation |
| `ml_train.yml` | ML training epochs + checkpoint management |