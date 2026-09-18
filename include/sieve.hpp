#ifndef RATPOINTS_GPU_SIEVE_HPP
#define RATPOINTS_GPU_SIEVE_HPP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "search_types.hpp"

namespace ratpoints_gpu {

struct SieveMetrics {
    float basis_ms = 0.0f;
    float sieve_ms = 0.0f;
    double plan_ms = 0.0;
    double workspace_ms = 0.0;
    double basis_wall_ms = 0.0;
    double survivor_setup_ms = 0.0;
    double verification_ms = 0.0;
    double sieve_wall_ms = 0.0;
    unsigned long long word_count = 0;
    long double initial_mask_bytes = 0.0L;
};

struct SieveResult {
    unsigned long long survivor_count = 0;
    SieveMetrics metrics;
};

struct CandidateBatch {
    std::vector<long long> numerators;
    std::vector<int> denominators;

    std::size_t size() const { return numerators.size(); }
};

using CandidateBatchCallback =
    std::function<void(const CandidateBatch &)>;

// The callback consumes each downloaded batch synchronously before GPU work
// resumes, keeping modular-candidate storage bounded.
SieveResult run_modular_sieve(const Coefficients &coefficients,
                               long long numerator_bound,
                               DenominatorRange denominators,
                               const CandidateBatchCallback &callback,
                               bool square_denominators = false);

}  // namespace ratpoints_gpu

#endif
