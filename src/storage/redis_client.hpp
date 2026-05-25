#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Redis Client
// Thin wrapper around hiredis for bar storage, state caching, pub/sub.
// Falls back to an in-memory map when hiredis is not available.
// ─────────────────────────────────────────────────────────────────────────────
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <stdexcept>
#include <memory>
#include <nlohmann/json.hpp>

#ifdef USE_REDIS
#include <hiredis/hiredis.h>
#endif

namespace dh::storage {

using json = nlohmann::json;

struct RedisConfig {
    std::string host    = "127.0.0.1";
    int         port    = 6379;
    int         db      = 0;
    std::string password;
    int         timeout_ms = 5000;
};

class RedisClient {
public:
    explicit RedisClient(RedisConfig cfg = {})
        : m_cfg(std::move(cfg)) {}

    ~RedisClient() { disconnect(); }

    bool connect();
    void disconnect();
    bool is_connected() const { return m_connected; }

    // ── Key/value ─────────────────────────────────────────────────────────────
    bool  set(const std::string& key, const std::string& value,
              int ttl_secs = 0);
    std::optional<std::string> get(const std::string& key);
    bool  del(const std::string& key);
    bool  exists(const std::string& key);

    // ── JSON helpers ──────────────────────────────────────────────────────────
    bool  set_json(const std::string& key, const json& val, int ttl_secs = 0);
    std::optional<json> get_json(const std::string& key);

    // ── List (timeseries stream) ───────────────────────────────────────────────
    int64_t lpush(const std::string& key, const std::string& val);
    int64_t rpush(const std::string& key, const std::string& val);
    std::vector<std::string> lrange(const std::string& key,
                                     int64_t start, int64_t stop);

    // ── Atomic counter ────────────────────────────────────────────────────────
    int64_t incr(const std::string& key);

    // ── Pub/Sub helpers ───────────────────────────────────────────────────────
    bool publish(const std::string& channel, const std::string& message);

private:
    RedisConfig m_cfg;
    bool        m_connected = false;

#ifdef USE_REDIS
    redisContext* m_ctx = nullptr;
    bool exec_cmd(const char* fmt, ...);
#else
    // In-memory fallback
    std::unordered_map<std::string, std::string>               m_store;
    std::unordered_map<std::string, std::vector<std::string>>  m_lists;
    std::unordered_map<std::string, int64_t>                   m_counters;
#endif
};

} // namespace dh::storage
