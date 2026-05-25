#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Hyper-Extensive Telemetry
// Emits structured JSON metrics compatible with TensorBoard / Prometheus.
// Covers: hparams, scalar metrics, histogram summaries, timing events.
// ─────────────────────────────────────────────────────────────────────────────
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <mutex>
#include <fstream>
#include <atomic>

namespace dh::telemetry {

using json = nlohmann::json;
using clock_t = std::chrono::steady_clock;
using time_point_t = clock_t::time_point;

// ── Scalar metric entry ───────────────────────────────────────────────────────
struct ScalarEntry {
    std::string tag;
    double      value;
    int64_t     step;
    int64_t     wall_time_ms;
};

// ── Histogram summary ─────────────────────────────────────────────────────────
struct HistogramEntry {
    std::string          tag;
    std::vector<double>  values;
    int64_t              step;
};

// ── HParam entry ──────────────────────────────────────────────────────────────
struct HParamEntry {
    std::string name;
    json        value;   // string | number | bool
};

// ─────────────────────────────────────────────────────────────────────────────
class Telemetry {
public:
    /// Singleton accessor.
    static Telemetry& instance();

    /// Initialise output directory and flush interval (seconds).
    void init(const std::string& output_dir, int flush_interval_secs = 10);

    // ── Scalars ───────────────────────────────────────────────────────────────
    void log_scalar(const std::string& tag, double value, int64_t step);
    void log_scalars(const std::unordered_map<std::string,double>& scalars,
                     int64_t step);

    // ── Histograms ────────────────────────────────────────────────────────────
    void log_histogram(const std::string& tag,
                       const std::vector<double>& values,
                       int64_t step);

    // ── HParams ──────────────────────────────────────────────────────────────
    void log_hparam(const std::string& name, const json& value);
    void log_hparams(const json& hparams_obj);

    // ── Timing events ────────────────────────────────────────────────────────
    void begin_event(const std::string& name);
    void end_event  (const std::string& name, int64_t step = -1);

    // ── Flush to disk ─────────────────────────────────────────────────────────
    void flush();

    // ── JSON snapshot (for checkpoint embedding) ──────────────────────────────
    json snapshot() const;

private:
    Telemetry() = default;

    mutable std::mutex                            m_mu;
    std::string                                   m_output_dir;
    std::ofstream                                 m_scalar_file;
    std::ofstream                                 m_hparam_file;
    std::ofstream                                 m_event_file;
    std::vector<ScalarEntry>                      m_scalars;
    std::vector<HistogramEntry>                   m_histograms;
    std::vector<HParamEntry>                      m_hparams;
    std::unordered_map<std::string,time_point_t>  m_open_events;
    std::atomic<bool>                             m_initialised{false};

    void flush_locked();
    int64_t now_ms() const;
};

// Convenience free-function wrappers
inline void scalar(const std::string& tag, double v, int64_t step)
{ Telemetry::instance().log_scalar(tag, v, step); }

inline void hparam(const std::string& name, const json& val)
{ Telemetry::instance().log_hparam(name, val); }

} // namespace dh::telemetry
