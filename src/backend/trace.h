#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

inline bool metalneedle_cpp_trace_enabled() {
  static bool enabled = std::getenv("METALNEEDLE_CPP_TRACE") != nullptr;
  return enabled;
}

struct TraceScope {
  std::string name;

  explicit TraceScope(std::string trace_name) : name(std::move(trace_name)) {
    if (metalneedle_cpp_trace_enabled()) {
      std::cerr << "[cpp] enter " << name << '\n';
    }
  }

  ~TraceScope() {
    if (metalneedle_cpp_trace_enabled()) {
      std::cerr << "[cpp] exit " << name << '\n';
    }
  }
};

#define TRACE_SCOPE(name) TraceScope trace_scope_instance(name)
