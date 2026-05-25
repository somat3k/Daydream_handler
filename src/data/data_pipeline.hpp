#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Data Pipeline
// Feeds OHLCV data from file / broker / replay into TimeSeries objects.
// ─────────────────────────────────────────────────────────────────────────────
#include "market_data.hpp"
#include "timeseries.hpp"
#include "../logging/logger.hpp"
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <string>
#include <fstream>
#include <vector>
#include <functional>
#include <stdexcept>

namespace dh::data {

using json = nlohmann::json;

// ── Feed source types ────────────────────────────────────────────────────────
enum class FeedSource { CSV_FILE, JSON_FILE, REDIS_STREAM, BROKER_LIVE };

struct PipelineConfig {
    FeedSource  source      = FeedSource::JSON_FILE;
    std::string data_path;          // file path or Redis key prefix
    size_t      window_size = 500;  // bars to keep in sliding window
    bool        replay_mode = true; // use LFO jitter replay
    LFOJitterParams lfo;
};

// ── Data Pipeline ─────────────────────────────────────────────────────────────
class DataPipeline {
public:
    using BarCallback = std::function<void(const std::string& symbol,
                                           Timeframe tf,
                                           const Bar& bar)>;

    explicit DataPipeline(PipelineConfig cfg)
        : m_cfg(std::move(cfg))
        , m_log(dh::logging::get("data_pipeline"))
    {}

    /// Register a callback invoked for every bar emitted by the pipeline.
    void on_bar(BarCallback cb) { m_callbacks.push_back(std::move(cb)); }

    /// Access the TimeSeries for a given symbol + timeframe.
    TimeSeries& series(const std::string& symbol, Timeframe tf) {
        auto key = make_key(symbol, tf);
        auto it  = m_series.find(key);
        if (it == m_series.end()) {
            m_series.emplace(key, TimeSeries(m_cfg.window_size));
            return m_series.at(key);
        }
        return it->second;
    }

    /// Load bars from a JSON file (array of bar objects).
    void load_json(const std::string& symbol, Timeframe tf,
                   const std::string& path)
    {
        m_log->info("Loading {} {} from {}", symbol,
                    static_cast<int>(tf), path);
        std::ifstream f(path);
        if (!f.is_open())
            throw std::runtime_error("Cannot open: " + path);
        json arr;
        f >> arr;
        auto& ts = series(symbol, tf);
        for (auto& j : arr) {
            Bar b{};
            b.time   = j.value("time",   int64_t{0});
            b.open   = j.value("open",   0.0);
            b.high   = j.value("high",   0.0);
            b.low    = j.value("low",    0.0);
            b.close  = j.value("close",  0.0);
            b.volume = j.value("volume", 0.0);
            b.tf     = tf;
            ts.push(b);
        }
        m_log->info("Loaded {} bars for {} TF{}", ts.size(), symbol,
                    static_cast<int>(tf));
    }

    /// Replay stored series through LFO jitter engine.
    void replay(const std::string& symbol, Timeframe tf)
    {
        LFOJitterReplay replayer(m_cfg.lfo);
        auto& ts  = series(symbol, tf);
        replayer.replay(ts, [&](const Bar& bar, int64_t idx){
            emit(symbol, tf, bar);
        }, m_cfg.replay_mode);
    }

    /// Push a single live bar (broker feed path).
    void push_bar(const std::string& symbol, Timeframe tf, const Bar& bar) {
        series(symbol, tf).push(bar);
        emit(symbol, tf, bar);
    }

private:
    static std::string make_key(const std::string& sym, Timeframe tf) {
        return sym + ":" + std::to_string(static_cast<int>(tf));
    }

    void emit(const std::string& sym, Timeframe tf, const Bar& bar) {
        for (auto& cb : m_callbacks) cb(sym, tf, bar);
    }

    PipelineConfig                             m_cfg;
    std::unordered_map<std::string,TimeSeries> m_series;
    std::vector<BarCallback>                   m_callbacks;
    std::shared_ptr<spdlog::logger>            m_log;
};

} // namespace dh::data
