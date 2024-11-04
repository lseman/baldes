#pragma once

#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

#include <fmt/color.h>
#include <fmt/core.h>

class Logger {
public:
    // Initialize the shared instance of Logger
    static void init(const std::string &file_name) {
        // Open in truncate mode to clear the file at the start
        instance().log_file.open(file_name, std::ios::out | std::ios::trunc);
        if (!instance().log_file.is_open()) { throw std::runtime_error("Could not open log file."); }
    }

    // Log function with static access
    template <typename... Args>
    static void log(fmt::format_string<Args...> format, Args &&...args) {
        // Write to the log file
        instance().log_file << fmt::format(format, std::forward<Args>(args)...);
    }

    // Log statistics with static access
    static void logStatistics(const std::string &statistics) { instance().log_file << statistics << std::endl; }

    // Close the log file when the program ends
    static void close() {
        if (instance().log_file.is_open()) { instance().log_file.close(); }
    }

private:
    // Private constructor
    Logger() = default;

    // Singleton pattern: Access to the single instance of Logger
    static Logger &instance() {
        static Logger logger_instance;
        return logger_instance;
    }

    std::ofstream log_file;
};
