#include "system_metrics.hpp"

#include <fstream>
#include <sstream>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <psapi.h>
#endif

namespace seg {

std::uint64_t current_rss_kb() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS_EX pmc{};
  pmc.cb = sizeof(pmc);
  if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc))) {
    return static_cast<std::uint64_t>(pmc.WorkingSetSize / 1024);
  }
  return 0;
#else
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      std::istringstream iss(line);
      std::string tag;
      std::uint64_t kb = 0;
      std::string unit;
      iss >> tag >> kb >> unit;
      return kb;
    }
  }
  return 0;
#endif
}

std::string gpu_utilization_line() {
#if defined(_WIN32)
  return "gpu: n/a (parse nvidia-smi on Jetson/Linux for utilization)";
#else
  return "gpu: run `tegrastats` on Jetson or `nvidia-smi` on dGPU hosts";
#endif
}

}  // namespace seg
