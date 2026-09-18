#include "sieve_plan.hpp"

#include <stdexcept>
#include <string>

namespace ratpoints_gpu {
namespace sieve_detail {
namespace {

static_assert(kInitialPrimeCount > 0 && kInitialPrimeCount <= kPrimeCount,
              "invalid initial sieve-prime count");
static_assert(kPrimeCount <= 93,
              "too many sieve primes for primes below 512");

int decimal_mod(const std::string &text, int modulus) {
    const char *digit = text.c_str();
    bool negative = *digit == '-';
    if (negative || *digit == '+') {
        ++digit;
    }
    int value = 0;
    while (*digit) {
        value = (10LL * value + (*digit++ - '0')) % modulus;
    }
    return negative && value ? modulus - value : value;
}

std::vector<int> primes_below(int limit) {
    std::vector<uint8_t> is_prime(limit, 1);
    std::vector<int> primes;
    is_prime[0] = is_prime[1] = 0;
    for (int candidate = 2; candidate < limit; ++candidate) {
        if (!is_prime[candidate]) {
            continue;
        }
        primes.push_back(candidate);
        if (static_cast<long long>(candidate) * candidate < limit) {
            for (int composite = candidate * candidate;
                 composite < limit; composite += candidate) {
                is_prime[composite] = 0;
            }
        }
    }
    return primes;
}

}  // namespace

SievePlan::SievePlan(long long numerator_bound, DenominatorRange range,
                     bool square_denominators)
    : denominator_range(range),
      square_denominators(square_denominators),
      numerator_bound(numerator_bound),
      word_count((2 * numerator_bound + 32) / 32) {
    if (range.first < 1 || range.first > range.last
        || (square_denominators && range.last > 46340)) {
        throw std::invalid_argument("invalid denominator range");
    }

    std::vector<int> all_primes = primes_below(kMaximumPrime);
    primes.assign(all_primes.end() - kCandidatePrimeCount, all_primes.end());
    if (primes.front() <= kBlockSize) {
        throw std::logic_error(
            "candidate primes must exceed the CUDA block size");
    }

    mask_offsets.resize(primes.size());
    for (size_t i = 0; i < primes.size(); ++i) {
        mask_offsets[i] = mask_count;
        mask_count += static_cast<long long>(primes[i]) * primes[i];
    }

    selected_primes.resize(static_cast<size_t>(range.count()) * kPrimeCount);
    // The inclusive endpoint can be INT_MAX; increment a wider cursor.
    for (long long parameter = range.first;
         parameter <= range.last; ++parameter) {
        long long denominator = square_denominators ? parameter * parameter : parameter;
        int selected_count = 0;
        size_t row = static_cast<size_t>(
            parameter - range.first) * kPrimeCount;
        for (int i = static_cast<int>(primes.size()) - 1;
             i >= 0 && selected_count < kPrimeCount; --i) {
            if (denominator % primes[i] != 0) {
                selected_primes[row + selected_count++] = i;
            }
        }
        if (selected_count != kPrimeCount) {
            throw std::logic_error("insufficient denominator sieve primes");
        }
    }

    for (int i = 0; i < kInitialPrimeCount; ++i) {
        shared_mask_bytes += primes[primes.size() - 1 - i]
            * sizeof(uint32_t);
    }
}

long long mod_inverse(long long value, long long modulus) {
    long long old_remainder = value;
    long long remainder = modulus;
    long long old_coefficient = 1;
    long long coefficient = 0;
    while (remainder) {
        long long quotient = old_remainder / remainder;
        long long next = old_remainder - quotient * remainder;
        old_remainder = remainder;
        remainder = next;
        next = old_coefficient - quotient * coefficient;
        old_coefficient = coefficient;
        coefficient = next;
    }
    return (old_coefficient % modulus + modulus) % modulus;
}

std::vector<uint8_t> build_affine_mask(
        int prime, const Coefficients &coefficients) {
    std::vector<int> modular_coefficients(coefficients.size());
    for (size_t i = 0; i < coefficients.size(); ++i) {
        modular_coefficients[i] = decimal_mod(coefficients[i], prime);
    }

    std::vector<uint8_t> is_square(prime, 0);
    for (int value = 0; value < prime; ++value) {
        is_square[static_cast<long long>(value) * value % prime] = 1;
    }

    std::vector<uint8_t> affine(prime, 0);
    for (int x = 0; x < prime; ++x) {
        int value = 0;
        for (int i = static_cast<int>(coefficients.size()) - 1;
             i >= 0; --i) {
            value = (static_cast<long long>(value) * x
                     + modular_coefficients[i]) % prime;
        }
        affine[x] = is_square[value];
    }
    return affine;
}

}  // namespace sieve_detail
}  // namespace ratpoints_gpu
