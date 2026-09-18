#ifndef RATPOINTS_GPU_SIEVE_PLAN_HPP
#define RATPOINTS_GPU_SIEVE_PLAN_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "search_types.hpp"

#ifndef NUM_PRIMES
#define NUM_PRIMES 32
#endif
#ifndef INITIAL_PRIMES
#if NUM_PRIMES < 14
#define INITIAL_PRIMES NUM_PRIMES
#else
#define INITIAL_PRIMES 14
#endif
#endif
#ifndef BLOCK
#define BLOCK 256
#endif

namespace ratpoints_gpu {
namespace sieve_detail {

constexpr int kPrimeCount = NUM_PRIMES;
constexpr int kInitialPrimeCount = INITIAL_PRIMES;
constexpr int kCandidatePrimeCount = kPrimeCount + 4;
constexpr int kMaximumPrime = 512;
constexpr int kBlockSize = BLOCK;

struct SievePlan {
    SievePlan(long long numerator_bound, DenominatorRange range,
              bool square_denominators = false);

    DenominatorRange denominator_range;
    bool square_denominators;
    long long numerator_bound;
    long long word_count;
    long long mask_count = 0;
    size_t shared_mask_bytes = 0;
    std::vector<int> primes;
    std::vector<long long> mask_offsets;
    std::vector<int> selected_primes;
};

long long mod_inverse(long long value, long long modulus);
std::vector<uint8_t> build_affine_mask(
    int prime, const Coefficients &coefficients);

}  // namespace sieve_detail
}  // namespace ratpoints_gpu

#endif
