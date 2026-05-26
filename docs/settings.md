# Daydream Handler – Settings & Configuration Reference

## Engine Configuration (`EngineConfig`)

```cpp
struct EngineConfig {
    // Instruments to process
    std::vector<std::string> symbols = {
        "XAUUSD", "XAUEUR", "XAGUSD", "XAGEUR",
        "GBPUSD", "EURUSD", "BTCUSD", "ETHUSD", "HYPEUSD"
    };

    // Timeframes to process
    std::vector<data::Timeframe> timeframes = {
        M1, M5, M15, H1, H4, D1
    };

    std::string model_dir      = "models";       // LibTorch model directory
    std::string checkpoint_dir = "checkpoints";  // Checkpoint save path
    std::string data_dir       = "data";         // Historical data path
    int    checkpoint_freq     = 1000;           // Bars between checkpoints
    bool   live_mode           = false;          // Live broker mode

    RiskConfig   risk;      // see below
    FeatureConfig features; // see below
};
```

---

## Risk Configuration (`RiskConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `account_equity` | `10000.0` | Account balance in account currency |
| `risk_pct` | `0.01` | Fraction of equity at risk per trade (1%) |
| `max_dd_pct` | `0.10` | Maximum tolerated drawdown (10%) |
| `min_tp_pips` | `1.0` | Minimum TP distance in pips |
| `max_tp_pips` | `80.0` | Maximum TP distance in pips |
| `max_spread_pips` | `3.0` | Reject signals when spread exceeds this |
| `max_positions` | `5.0` | Maximum concurrent open positions |

---

## Feature Configuration (`FeatureConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window` | `50` | Lookback bars for OHLCV timeseries features |
| `normalise` | `true` | Apply Z-score normalisation to price windows |
| `include_ich` | `true` | Include Ichimoku values in feature vector |
| `include_vol` | `true` | Include Volatility Cloud values |
| `include_fib` | `true` | Include Fibonacci levels |
| `project_dim` | `64` | GEMM projection output dimension (0 = skip) |

---

## Indicator Parameters

### Ichimoku (`IchimokuParams`)
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `tenkan_period` | 9 | Conversion line period |
| `kijun_period` | 26 | Base line period |
| `senkou_b_period` | 52 | Leading Span B period |
| `displacement` | 26 | Forward displacement |

### Volatility Cloud (`VolatilityCloud::Params`)
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `atr_period` | 14 | ATR calculation period |
| `multiplier` | 1.5 | Band = mid ± mult × ATR |
| `smooth_period` | 5 | EMA smoothing for midpoint |

### Supply & Demand (`SupplyDemand::Params`)
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `impulse_bars` | 3 | Minimum bars in impulse leg |
| `impulse_ratio` | 0.003 | Minimum move as fraction of price |
| `base_bars` | 5 | Consolidation zone width |
| `max_zones` | 20 | Maximum zones to detect |

### LFO Jitter (`LFOJitterParams`)
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `frequency` | 0.01 | Cycles per bar |
| `amplitude` | 0.0002 | Jitter amplitude as fraction of price |
| `phase` | 0.0 | Initial phase in radians |
| `add_noise` | true | Add Gaussian noise on top of LFO |
| `noise_sigma` | 0.0001 | Noise std as fraction of price |
| `seed` | 42 | RNG seed for reproducibility |

---

## Strategy Parameters

### Breakout (`BreakoutStrategy::Params`)
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `lookback` | 20 | Bars for range detection |
| `threshold` | 0.001 | Confirmation beyond range (0.1%) |

### StrategyMultiplexer Config
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `min_confidence` | 0.55 | Minimum blended signal confidence |
| `require_ml_agree` | true | ML model must confirm strategy direction |
| `require_dt_gate` | true | Decision tree pre-filter must pass |

---

## Redis Configuration (`RedisConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `127.0.0.1` | Redis server hostname |
| `port` | `6379` | Redis server port |
| `db` | `0` | Redis database index |
| `password` | `""` | Redis AUTH password (use GitHub Secret `REDIS_PASSWORD`) |
| `timeout_ms` | `5000` | Connection timeout in milliseconds |

---

## Logging Configuration

Controlled via `dh::logging::init()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_dir` | `"logs"` | Directory for log files |
| `log_name` | `"daydream"` | Base filename prefix |
| `console_level` | `info` | Console output verbosity |
| `file_level` | `trace` | File log verbosity |

Log levels: `trace < debug < info < warn < error < critical`

---

## Telemetry Configuration

Controlled via `Telemetry::init()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `output_dir` | `"logs/telemetry"` | Telemetry output directory |
| `flush_interval_secs` | `10` | Auto-flush interval (future) |

---

## CLI Options

```
daydream_handler <command> [options]

Commands:
  replay      Run LFO jitter replay over historical data
  live        Live broker feed (requires broker plugin)
  info        Print model registry and instrument specs
  checkpoint  List available checkpoints
  help        Show usage

Options:
  -d, --data-dir    <path>   Historical data directory    [data]
  -m, --model-dir   <path>   ML model directory           [models]
  -k, --ckpt-dir    <path>   Checkpoint directory         [checkpoints]
  -l, --log-dir     <path>   Log output directory         [logs]
  -s, --symbol      <sym>    Instrument symbol            [XAUUSD]
  -t, --timeframe   <tf>     M1|M5|M15|H1|H4|D1          [H1]
  -e, --equity      <float>  Account equity               [10000]
  -r, --risk        <float>  Risk fraction per trade      [0.01]
  -n, --epochs      <int>    Replay epochs                [1]
      --live                 Enable live mode
  -v, --verbose              Verbose console output
```

---

## Build Options (CMake)

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_TESTS` | `ON` | Build unit and integration tests |
| `ENABLE_TELEMETRY` | `ON` | Enable hyper-extensive telemetry |
| `USE_LIBTORCH` | `ON` | Link LibTorch for TorchScript `.pt` ML inference |
| `USE_ONNXRUNTIME` | `OFF` | Link ONNX Runtime for native `.onnx` ML inference |
| `USE_REDIS` | `ON` | Link hiredis for Redis storage |
| `CMAKE_BUILD_TYPE` | `Release` | `Release` / `Debug` / `RelWithDebInfo` |

---

## GitHub Secrets

Configure at `repo → Settings → Secrets and variables → Actions`:

| Secret Name | Required | Purpose |
|-------------|----------|---------|
| `BROKER_API_KEY` | Future | Live broker API key |
| `BROKER_API_SECRET` | Future | Live broker API secret |
| `REDIS_PASSWORD` | Optional | Production Redis auth |
| `HF_TOKEN` | Optional | HuggingFace model download |
