#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <gmpxx.h>

#include "sieve.hpp"

namespace {

constexpr long long kBound = 5000;
constexpr int kLastDenominator = 2;
bool is_prime(int value) {
    for (int divisor = 2; divisor * divisor <= value; ++divisor) {
        if (value % divisor == 0) {
            return false;
        }
    }
    return value >= 2;
}

ratpoints_gpu::Coefficients all_surviving_polynomial() {
    std::vector<int> primes;
    for (int value = 2; value < 512; ++value) {
        if (is_prime(value)) {
            primes.push_back(value);
        }
    }
    mpz_class product = 1;
    for (auto prime = primes.end() - 32; prime != primes.end(); ++prime) {
        product *= *prime;
    }
    return {"1", product.get_str()};
}

size_t candidate_index(long long numerator, int denominator) {
    size_t row = static_cast<size_t>(denominator - 1);
    size_t column = static_cast<size_t>(numerator + kBound);
    return row * static_cast<size_t>(2 * kBound + 1) + column;
}

}  // namespace

int main() {
    const size_t expected_count = static_cast<size_t>(kLastDenominator)
        * static_cast<size_t>(2 * kBound + 1);
    std::vector<bool> seen(expected_count, false);
    size_t callback_count = 0;

    ratpoints_gpu::SieveResult result = ratpoints_gpu::run_modular_sieve(
        all_surviving_polynomial(), kBound, {1, kLastDenominator},
        [&](const ratpoints_gpu::CandidateBatch &batch) {
            if (batch.denominators.size() != batch.size()) {
                throw std::runtime_error("invalid candidate batch size");
            }
            callback_count += batch.size();
            for (size_t i = 0; i < batch.size(); ++i) {
                long long numerator = batch.numerators[i];
                int denominator = batch.denominators[i];
                if (numerator < -kBound || numerator > kBound
                    || denominator < 1
                    || denominator > kLastDenominator) {
                    throw std::runtime_error("candidate outside search box");
                }
                size_t index = candidate_index(numerator, denominator);
                if (seen[index]) {
                    throw std::runtime_error("duplicate modular candidate");
                }
                seen[index] = true;
            }
        });

    if (result.survivor_count != expected_count
        || callback_count != expected_count
        || !std::all_of(seen.begin(), seen.end(), [](bool value) {
               return value;
           })) {
        throw std::runtime_error("incomplete modular candidate stream");
    }
}
