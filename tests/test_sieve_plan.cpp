#include <climits>
#include <iostream>
#include <stdexcept>

#include "sieve_plan.hpp"

int main() {
    using namespace ratpoints_gpu;
    using namespace ratpoints_gpu::sieve_detail;
    for (DenominatorRange range : {DenominatorRange{1, 1}, {65535, 65538},
                                   {INT_MAX - 2, INT_MAX}}) {
        SievePlan plan(100, range);
        if (plan.selected_primes.size()
            != static_cast<size_t>(range.count()) * kPrimeCount) {
            throw std::runtime_error("incorrect sieve-plan row count");
        }
        for (long long b = range.first; b <= range.last; ++b) {
            for (int i = 0; i < kPrimeCount; ++i) {
                size_t offset = static_cast<size_t>(b - range.first) * kPrimeCount + i;
                int prime = plan.primes.at(plan.selected_primes.at(offset));
                if (b % prime == 0) {
                    throw std::runtime_error("selected a forbidden divisor");
                }
            }
        }
    }
    std::cout << "Sieve-plan boundary tests passed (real host implementation).\n";
}
