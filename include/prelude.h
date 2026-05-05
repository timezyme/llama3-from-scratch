// Project-wide using-declarations and type aliases.
//
// Pulled into the global namespace deliberately to keep call sites
// terse (e.g. `vector<int>` instead of `std::vector<int>` everywhere
// in tokenizer.h, loader.h, etc.). This is acceptable for an
// application-only project — for a public library we'd keep names
// fully qualified.

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
