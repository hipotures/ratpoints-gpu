#include "exact_polynomial.hpp"

#include <utility>

namespace ratpoints_gpu {
namespace {

using RationalPolynomial = std::vector<mpq_class>;

void trim(RationalPolynomial &polynomial) {
    while (!polynomial.empty() && polynomial.back() == 0) {
        polynomial.pop_back();
    }
}

}  // namespace

ExactPolynomial::ExactPolynomial(const Coefficients &coefficients) {
    coefficients_.reserve(coefficients.size());
    for (const auto &coefficient : coefficients) {
        coefficients_.emplace_back(coefficient, 10);
    }
}

bool ExactPolynomial::square_root(long long numerator, int denominator,
                                  mpz_class &root) const {
    mpz_class value = coefficients_.back();
    mpz_class power = 1;
    for (int i = static_cast<int>(coefficients_.size()) - 2; i >= 0; --i) {
        mpz_mul_si(value.get_mpz_t(), value.get_mpz_t(), numerator);
        mpz_mul_si(power.get_mpz_t(), power.get_mpz_t(), denominator);
        value += coefficients_[i] * power;
    }
    if (coefficients_.size() % 2 == 0) {
        mpz_mul_si(value.get_mpz_t(), value.get_mpz_t(), denominator);
    }
    if (value < 0 || !mpz_perfect_square_p(value.get_mpz_t())) {
        return false;
    }
    mpz_sqrt(root.get_mpz_t(), value.get_mpz_t());
    return true;
}

bool ExactPolynomial::is_squarefree(const Coefficients &coefficients) {
    RationalPolynomial polynomial;
    polynomial.reserve(coefficients.size());
    for (const auto &coefficient : coefficients) {
        polynomial.emplace_back(mpz_class(coefficient, 10));
    }

    RationalPolynomial derivative(polynomial.size() - 1);
    for (size_t i = 1; i < polynomial.size(); ++i) {
        derivative[i - 1] = polynomial[i] * i;
    }
    trim(derivative);

    while (!derivative.empty()) {
        RationalPolynomial remainder = polynomial;
        while (remainder.size() >= derivative.size()) {
            size_t shift = remainder.size() - derivative.size();
            mpq_class factor = remainder.back() / derivative.back();
            for (size_t i = 0; i < derivative.size(); ++i) {
                remainder[i + shift] -= factor * derivative[i];
            }
            trim(remainder);
        }
        polynomial = std::move(derivative);
        derivative = std::move(remainder);
    }
    return polynomial.size() == 1;
}

}  // namespace ratpoints_gpu
