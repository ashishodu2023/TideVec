// ================================================================
// exceptions.hpp — CortexDB C++ SDK exceptions
// ================================================================

#pragma once

#include <stdexcept>
#include <string>

namespace cortexdb {

struct CortexDBError : std::runtime_error {
    explicit CortexDBError(const std::string& msg)
        : std::runtime_error("CortexDB: " + msg) {}
};

struct ConnectionError : CortexDBError {
    explicit ConnectionError(const std::string& host)
        : CortexDBError("Cannot connect to server at " + host) {}
};

struct CollectionNotFound : CortexDBError {
    explicit CollectionNotFound(const std::string& name)
        : CortexDBError("Collection not found: " + name) {}
};

struct DimensionMismatch : CortexDBError {
    DimensionMismatch(int expected, int got)
        : CortexDBError("Dimension mismatch: expected " +
                        std::to_string(expected) + ", got " +
                        std::to_string(got)) {}
};

struct AuthError : CortexDBError {
    explicit AuthError()
        : CortexDBError("Authentication failed — check your API key") {}
};

} // namespace cortexdb
