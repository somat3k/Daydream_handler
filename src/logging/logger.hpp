#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Daydream Handler – Logger
// Wraps spdlog with named loggers and structured JSON sink support.
// ─────────────────────────────────────────────────────────────────────────────
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <memory>
#include <string>

namespace dh::logging {

/// Central factory – call once at program start.
void init(const std::string& log_dir  = "logs",
          const std::string& log_name = "daydream",
          spdlog::level::level_enum console_level = spdlog::level::info,
          spdlog::level::level_enum file_level    = spdlog::level::trace);

/// Returns (or creates) a named child logger.
std::shared_ptr<spdlog::logger> get(const std::string& name);

/// Flush all sinks.
void flush_all();

/// Singleton access to the root logger.
std::shared_ptr<spdlog::logger> root();

} // namespace dh::logging
