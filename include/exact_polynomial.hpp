#ifndef RATPOINTS_GPU_EXACT_POLYNOMIAL_HPP
#define RATPOINTS_GPU_EXACT_POLYNOMIAL_HPP

#include <gmpxx.h>
#include <vector>

#include "search_types.hpp"

namespace ratpoints_gpu {

class ExactPolynomial {
public:
    explicit ExactPolynomial(const Coefficients &coefficients);

    bool square_root(long long numerator, int denominator,
                     mpz_class &root) const;
    static bool is_squarefree(const Coefficients &coefficients);

private:
    std::vector<mpz_class> coefficients_;
};

}  // namespace ratpoints_gpu

#endif
