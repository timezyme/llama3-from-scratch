// Common type aliases and STL imports shared across all project headers.
// Included by nearly every file to avoid repetitive using-declarations.

#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Bring frequently used STL types into the global namespace for brevity.
using std::runtime_error;
using std::string;
using std::unordered_map;
using std::vector;

// Project-wide numeric type aliases.
using float_t = float;
using size_t = std::size_t;
