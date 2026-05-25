#include "redis_client.hpp"
#include <sstream>

namespace dh::storage {

// ── In-memory fallback implementation (always compiled) ──────────────────────
// When USE_REDIS is defined the hiredis backend is used instead.

bool RedisClient::connect()
{
#ifdef USE_REDIS
    struct timeval tv{0, m_cfg.timeout_ms * 1000};
    m_ctx = redisConnectWithTimeout(
        m_cfg.host.c_str(), m_cfg.port, tv);
    if (!m_ctx || m_ctx->err) {
        m_connected = false;
        return false;
    }
    if (!m_cfg.password.empty()) {
        redisReply* r = static_cast<redisReply*>(
            redisCommand(m_ctx, "AUTH %s", m_cfg.password.c_str()));
        bool ok = r && r->type != REDIS_REPLY_ERROR;
        if (r) freeReplyObject(r);
        if (!ok) return false;
    }
    if (m_cfg.db != 0) {
        redisReply* r = static_cast<redisReply*>(
            redisCommand(m_ctx, "SELECT %d", m_cfg.db));
        bool ok = r && r->type != REDIS_REPLY_ERROR;
        if (r) freeReplyObject(r);
        if (!ok) return false;
    }
    m_connected = true;
    return true;
#else
    m_connected = true;  // in-memory always "connected"
    return true;
#endif
}

void RedisClient::disconnect()
{
#ifdef USE_REDIS
    if (m_ctx) { redisFree(m_ctx); m_ctx = nullptr; }
#endif
    m_connected = false;
}

bool RedisClient::set(const std::string& key, const std::string& value,
                      int ttl_secs)
{
#ifdef USE_REDIS
    redisReply* r = ttl_secs > 0
        ? static_cast<redisReply*>(redisCommand(m_ctx, "SETEX %s %d %s",
              key.c_str(), ttl_secs, value.c_str()))
        : static_cast<redisReply*>(redisCommand(m_ctx, "SET %s %s",
              key.c_str(), value.c_str()));
    bool ok = r && r->type == REDIS_REPLY_STATUS;
    if (r) freeReplyObject(r);
    return ok;
#else
    (void)ttl_secs;
    m_store[key] = value;
    return true;
#endif
}

std::optional<std::string> RedisClient::get(const std::string& key)
{
#ifdef USE_REDIS
    redisReply* r = static_cast<redisReply*>(
        redisCommand(m_ctx, "GET %s", key.c_str()));
    std::optional<std::string> result;
    if (r && r->type == REDIS_REPLY_STRING)
        result = std::string(r->str, r->len);
    if (r) freeReplyObject(r);
    return result;
#else
    auto it = m_store.find(key);
    if (it == m_store.end()) return {};
    return it->second;
#endif
}

bool RedisClient::del(const std::string& key)
{
#ifdef USE_REDIS
    redisReply* r = static_cast<redisReply*>(
        redisCommand(m_ctx, "DEL %s", key.c_str()));
    bool ok = r && r->integer > 0;
    if (r) freeReplyObject(r);
    return ok;
#else
    return m_store.erase(key) > 0;
#endif
}

bool RedisClient::exists(const std::string& key)
{
#ifdef USE_REDIS
    redisReply* r = static_cast<redisReply*>(
        redisCommand(m_ctx, "EXISTS %s", key.c_str()));
    bool ok = r && r->integer > 0;
    if (r) freeReplyObject(r);
    return ok;
#else
    return m_store.count(key) > 0;
#endif
}

bool RedisClient::set_json(const std::string& key, const json& val,
                            int ttl_secs)
{
    return set(key, val.dump(), ttl_secs);
}

std::optional<json> RedisClient::get_json(const std::string& key)
{
    auto raw = get(key);
    if (!raw) return {};
    try { return json::parse(*raw); }
    catch (...) { return {}; }
}

int64_t RedisClient::lpush(const std::string& key, const std::string& val)
{
#ifdef USE_REDIS
    redisReply* r = static_cast<redisReply*>(
        redisCommand(m_ctx, "LPUSH %s %s", key.c_str(), val.c_str()));
    int64_t n = r ? r->integer : 0;
    if (r) freeReplyObject(r);
    return n;
#else
    m_lists[key].insert(m_lists[key].begin(), val);
    return static_cast<int64_t>(m_lists[key].size());
#endif
}

int64_t RedisClient::rpush(const std::string& key, const std::string& val)
{
#ifdef USE_REDIS
    redisReply* r = static_cast<redisReply*>(
        redisCommand(m_ctx, "RPUSH %s %s", key.c_str(), val.c_str()));
    int64_t n = r ? r->integer : 0;
    if (r) freeReplyObject(r);
    return n;
#else
    m_lists[key].push_back(val);
    return static_cast<int64_t>(m_lists[key].size());
#endif
}

std::vector<std::string> RedisClient::lrange(const std::string& key,
                                               int64_t start, int64_t stop)
{
#ifdef USE_REDIS
    redisReply* r = static_cast<redisReply*>(
        redisCommand(m_ctx, "LRANGE %s %lld %lld",
                     key.c_str(), (long long)start, (long long)stop));
    std::vector<std::string> result;
    if (r && r->type == REDIS_REPLY_ARRAY)
        for (size_t i = 0; i < r->elements; ++i)
            result.emplace_back(r->element[i]->str, r->element[i]->len);
    if (r) freeReplyObject(r);
    return result;
#else
    auto it = m_lists.find(key);
    if (it == m_lists.end()) return {};
    auto& v = it->second;
    int64_t sz = static_cast<int64_t>(v.size());
    if (start < 0) start = std::max(int64_t{0}, sz + start);
    if (stop  < 0) stop  = sz + stop;
    stop = std::min(stop, sz - 1);
    if (start > stop) return {};
    return std::vector<std::string>(
        v.begin() + start, v.begin() + stop + 1);
#endif
}

int64_t RedisClient::incr(const std::string& key)
{
#ifdef USE_REDIS
    redisReply* r = static_cast<redisReply*>(
        redisCommand(m_ctx, "INCR %s", key.c_str()));
    int64_t n = r ? r->integer : 0;
    if (r) freeReplyObject(r);
    return n;
#else
    return ++m_counters[key];
#endif
}

bool RedisClient::publish(const std::string& channel,
                           const std::string& message)
{
#ifdef USE_REDIS
    redisReply* r = static_cast<redisReply*>(
        redisCommand(m_ctx, "PUBLISH %s %s",
                     channel.c_str(), message.c_str()));
    bool ok = r != nullptr;
    if (r) freeReplyObject(r);
    return ok;
#else
    (void)channel; (void)message;
    return true;  // in-memory pub/sub is a no-op
#endif
}

} // namespace dh::storage
