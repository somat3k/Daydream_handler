#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Market Data Types
// OHLCV bars, tick data, instrument metadata.
// ─────────────────────────────────────────────────────────────────────────────
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>

namespace dh::data {

using Timestamp = int64_t;  // Unix ms

// ── Instrument ────────────────────────────────────────────────────────────────
enum class Timeframe : uint32_t {
    M1  = 1,
    M5  = 5,
    M15 = 15,
    M30 = 30,
    H1  = 60,
    H4  = 240,
    D1  = 1440,
    W1  = 10080,
    MN  = 43200,
};

struct Instrument {
    std::string symbol;     // e.g. "XAUUSD"
    double      pip_size;   // e.g. 0.01 for XAUUSD
    double      lot_size;   // base contract size
    double      commission; // per-lot commission in account currency
    double      spread;     // typical spread in pips
};

// ── OHLCV Bar ─────────────────────────────────────────────────────────────────
struct Bar {
    Timestamp time;   // bar open time (Unix ms)
    double    open;
    double    high;
    double    low;
    double    close;
    double    volume;
    Timeframe tf;
};

// ── Tick ──────────────────────────────────────────────────────────────────────
struct Tick {
    Timestamp time;
    double    bid;
    double    ask;
    double    last;
    double    volume;
};

// ── Multi-timeframe bar set ───────────────────────────────────────────────────
struct MTFBars {
    std::vector<Bar> m1;
    std::vector<Bar> m5;
    std::vector<Bar> m15;
    std::vector<Bar> m30;
    std::vector<Bar> h1;
    std::vector<Bar> h4;
    std::vector<Bar> d1;
};

// Common instrument definitions
namespace instruments {

inline Instrument XAUUSD() { return {"XAUUSD", 0.01, 100.0,  0.05, 0.3};  }
inline Instrument XAUEUR() { return {"XAUEUR", 0.01, 100.0,  0.05, 0.4};  }
inline Instrument XAGUSD() { return {"XAGUSD", 0.001,5000.0, 0.02, 0.5};  }
inline Instrument XAGEUR() { return {"XAGEUR", 0.001,5000.0, 0.02, 0.6};  }
inline Instrument GBPUSD() { return {"GBPUSD", 0.0001,100000.0,3.0, 0.8}; }
inline Instrument EURUSD() { return {"EURUSD", 0.0001,100000.0,2.0, 0.5}; }
inline Instrument BTCUSD() { return {"BTCUSD", 0.01, 1.0,     5.0, 2.0};  }
inline Instrument ETHUSD() { return {"ETHUSD", 0.01, 1.0,     3.0, 1.5};  }
inline Instrument HYPEUSD(){ return {"HYPEUSD",0.001,1.0,     2.0, 1.0};  }

} // namespace instruments

} // namespace dh::data
