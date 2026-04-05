#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

namespace seg {

class Logger {
 public:
  explicit Logger(const std::string& path) : file_(path, std::ios::out | std::ios::app) {}

  void info(const std::string& msg) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto line = ts() + " [INFO] " + msg;
    std::cout << line << std::endl;
    if (file_.is_open()) {
      file_ << line << std::endl;
    }
  }

  void warn(const std::string& msg) {
    std::lock_guard<std::mutex> lock(mu_);
    const auto line = ts() + " [WARN] " + msg;
    std::cerr << line << std::endl;
    if (file_.is_open()) {
      file_ << line << std::endl;
    }
  }

 private:
  static std::string ts() {
    using clock = std::chrono::system_clock;
    const auto t = clock::now();
    const auto tt = clock::to_time_t(t);
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_buf, "%Y-%m-%d %H:%M:%S");
    return oss.str();
  }

  std::ofstream file_;
  std::mutex mu_;
};

}  // namespace seg
