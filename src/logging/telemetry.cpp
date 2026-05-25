#include "telemetry.hpp"
#include <filesystem>
#include <numeric>
#include <cmath>
#include <iomanip>

namespace dh::telemetry {

Telemetry& Telemetry::instance()
{
    static Telemetry inst;
    return inst;
}

void Telemetry::init(const std::string& output_dir, int /*flush_interval_secs*/)
{
    std::lock_guard lock(m_mu);
    if (m_initialised.load()) return;
    m_output_dir = output_dir;
    std::filesystem::create_directories(output_dir);
    m_scalar_file.open(output_dir + "/scalars.jsonl", std::ios::app);
    m_hparam_file.open(output_dir + "/hparams.json",  std::ios::trunc);
    m_event_file .open(output_dir + "/events.jsonl",  std::ios::app);
    m_initialised.store(true);
}

int64_t Telemetry::now_ms() const
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

void Telemetry::log_scalar(const std::string& tag, double value, int64_t step)
{
    std::lock_guard lock(m_mu);
    m_scalars.push_back({tag, value, step, now_ms()});
}

void Telemetry::log_scalars(const std::unordered_map<std::string,double>& scalars,
                             int64_t step)
{
    for (auto& [k,v] : scalars) log_scalar(k, v, step);
}

void Telemetry::log_histogram(const std::string& tag,
                               const std::vector<double>& values,
                               int64_t step)
{
    std::lock_guard lock(m_mu);
    m_histograms.push_back({tag, values, step});
}

void Telemetry::log_hparam(const std::string& name, const json& value)
{
    std::lock_guard lock(m_mu);
    m_hparams.push_back({name, value});
}

void Telemetry::log_hparams(const json& hparams_obj)
{
    for (auto& [k,v] : hparams_obj.items()) log_hparam(k, v);
}

void Telemetry::begin_event(const std::string& name)
{
    std::lock_guard lock(m_mu);
    m_open_events[name] = clock_t::now();
}

void Telemetry::end_event(const std::string& name, int64_t step)
{
    std::lock_guard lock(m_mu);
    auto it = m_open_events.find(name);
    if (it == m_open_events.end()) return;
    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        clock_t::now() - it->second).count();
    m_open_events.erase(it);

    if (m_event_file.is_open()) {
        json ev = {
            {"name",       name},
            {"elapsed_us", elapsed_us},
            {"step",       step},
            {"wall_ms",    now_ms()}
        };
        m_event_file << ev.dump() << "\n";
    }
    // Also record as scalar for chart visualisation
    m_scalars.push_back({name + "/elapsed_us",
                         static_cast<double>(elapsed_us), step, now_ms()});
}

void Telemetry::flush()
{
    std::lock_guard lock(m_mu);
    flush_locked();
}

void Telemetry::flush_locked()
{
    if (!m_initialised.load()) return;

    // Write scalars
    for (auto& s : m_scalars) {
        if (!m_scalar_file.is_open()) break;
        json j = {
            {"tag",       s.tag},
            {"value",     s.value},
            {"step",      s.step},
            {"wall_ms",   s.wall_time_ms}
        };
        m_scalar_file << j.dump() << "\n";
    }
    m_scalars.clear();

    // Write histograms inline into scalar file as summary stats
    for (auto& h : m_histograms) {
        if (h.values.empty() || !m_scalar_file.is_open()) continue;
        double sum  = std::accumulate(h.values.begin(), h.values.end(), 0.0);
        double mean = sum / static_cast<double>(h.values.size());
        double sq_sum = std::inner_product(
            h.values.begin(), h.values.end(), h.values.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / h.values.size() - mean * mean);
        auto   minmax = std::minmax_element(h.values.begin(), h.values.end());

        json j = {
            {"tag",    h.tag + "/hist"},
            {"mean",   mean},
            {"stdev",  stdev},
            {"min",    *minmax.first},
            {"max",    *minmax.second},
            {"count",  h.values.size()},
            {"step",   h.step},
            {"wall_ms",now_ms()}
        };
        m_scalar_file << j.dump() << "\n";
    }
    m_histograms.clear();

    // Write hparams
    if (m_hparam_file.is_open() && !m_hparams.empty()) {
        json hp = json::object();
        for (auto& p : m_hparams) hp[p.name] = p.value;
        m_hparam_file.seekp(0);
        m_hparam_file << hp.dump(2) << "\n";
        m_hparam_file.flush();
    }

    if (m_scalar_file.is_open()) m_scalar_file.flush();
    if (m_event_file.is_open())  m_event_file.flush();
}

json Telemetry::snapshot() const
{
    std::lock_guard lock(m_mu);
    json hp = json::object();
    for (auto& p : m_hparams) hp[p.name] = p.value;
    return {{"hparams", hp}, {"pending_scalars", m_scalars.size()}};
}

} // namespace dh::telemetry
