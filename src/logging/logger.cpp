#include "logger.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <filesystem>
#include <mutex>

namespace dh::logging {

namespace {
    std::shared_ptr<spdlog::logger> s_root;
    std::mutex s_mu;
}

void init(const std::string& log_dir,
          const std::string& log_name,
          spdlog::level::level_enum console_level,
          spdlog::level::level_enum file_level)
{
    std::lock_guard lock(s_mu);
    if (s_root) return;  // idempotent

    std::filesystem::create_directories(log_dir);

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(console_level);
    console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");

    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        log_dir + "/" + log_name + ".log",
        /* max_size */ 50 * 1024 * 1024,  // 50 MB
        /* max_files */ 10
    );
    file_sink->set_level(file_level);
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] [%s:%#] %v");

    s_root = std::make_shared<spdlog::logger>(
        log_name,
        spdlog::sinks_init_list{console_sink, file_sink}
    );
    s_root->set_level(spdlog::level::trace);
    spdlog::register_logger(s_root);
    spdlog::set_default_logger(s_root);
}

std::shared_ptr<spdlog::logger> get(const std::string& name)
{
    auto existing = spdlog::get(name);
    if (existing) return existing;

    auto parent = root();
    auto child  = parent->clone(name);
    spdlog::register_logger(child);
    return child;
}

void flush_all()
{
    spdlog::apply_all([](const std::shared_ptr<spdlog::logger>& l){ l->flush(); });
}

std::shared_ptr<spdlog::logger> root()
{
    std::lock_guard lock(s_mu);
    if (!s_root) {
        // fallback: console-only initialisation
        s_root = spdlog::stdout_color_mt("daydream");
    }
    return s_root;
}

} // namespace dh::logging
