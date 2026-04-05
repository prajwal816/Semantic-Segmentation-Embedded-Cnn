#pragma once

#include <cstdint>
#include <string>

namespace seg {

std::uint64_t current_rss_kb();

std::string gpu_utilization_line();

}  // namespace seg
