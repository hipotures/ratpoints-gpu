#ifndef RATPOINTS_GPU_SIEVE_HPP
#define RATPOINTS_GPU_SIEVE_HPP

#include <cstdint>
#include <vector>

#include "search_types.hpp"

namespace ratpoints_gpu {

struct SieveMetrics {
    float basis_ms = 0.0f;
    float sieve_ms = 0.0f;
    unsigned long long word_count = 0;
    long double initial_mask_bytes = 0.0L;
};

struct SieveResult {
    std::vector<ModularCandidate> candidates;
    SieveMetrics metrics;
};

SieveResult run_modular_sieve(const Coefficients &coefficients,
                              long long numerator_bound,
                              DenominatorRange denominators);

}  // namespace ratpoints_gpu

#endif
