// ================================================================
// exceptions.hpp — TideVec C++ SDK exceptions
// ================================================================

#pragma once

#include <stdexcept>
#include <string>

namespace tidevec {

struct TideVecError : std::runtime_error {
    explicit TideVecError(const std::string& msg)
        : std::runtime_error("TideVec: " + msg) {}
};

struct ConnectionError : TideVecError {
    explicit ConnectionError(const std::string& host)
        : TideVecError("Cannot connect to server at " + host) {}
};

struct CollectionNotFound : TideVecError {
    explicit CollectionNotFound(const std::string& name)
        : TideVecError("Collection not found: " + name) {}
};

struct DimensionMismatch : TideVecError {
    DimensionMismatch(int expected, int got)
        : TideVecError("Dimension mismatch: expected " +
                        std::to_string(expected) + ", got " +
                        std::to_string(got)) {}
};

struct AuthError : TideVecError {
    explicit AuthError()
        : TideVecError("Authentication failed — check your API key") {}
};

} // namespace tidevec
