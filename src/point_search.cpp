#include "point_search.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "exact_polynomial.hpp"
#include "sieve.hpp"

namespace ratpoints_gpu {
namespace {

constexpr int kDenominatorBatchSize = 1 << 16;

SearchOptions validate_search_options(SearchOptions options) {
    if (!ExactPolynomial::is_squarefree(options.coefficients)) {
        throw std::invalid_argument("polynomial is not squarefree");
    }
    return options;
}

long long integer_gcd(long long a, long long b) {
    if (a < 0) {
        a = -a;
    }
    while (b) {
        long long next = a % b;
        a = b;
        b = next;
    }
    return a;
}

void add_checked(unsigned long long &total, unsigned long long value,
                 const char *message) {
    if (value > std::numeric_limits<unsigned long long>::max() - total) {
        throw std::overflow_error(message);
    }
    total += value;
}

void append_verified_candidates(
        const CandidateBatch &candidates,
        const ExactPolynomial &polynomial, std::vector<PointPair> &points) {
    mpz_class root;
    for (size_t i = 0; i < candidates.size(); ++i) {
        long long numerator = candidates.numerators[i];
        int denominator = candidates.denominators[i];
        if (integer_gcd(numerator, denominator) == 1
            && polynomial.square_root(numerator, denominator, root)) {
            points.push_back(
                {numerator, root.get_str(), denominator});
        }
    }
}

void sort_points(std::vector<PointPair> &points) {
    std::sort(points.begin(), points.end(),
              [](const PointPair &left, const PointPair &right) {
                  if (left.denominator != right.denominator) {
                      return left.denominator < right.denominator;
                  }
                  return left.numerator < right.numerator;
              });
}

}  // namespace

bool SearchOptions::accepts(long long numerator, int denominator) const {
    if (intervals.empty()) {
        return true;
    }
    mpq_class x(std::to_string(numerator));
    x /= denominator;
    x.canonicalize();
    for (const auto &interval : intervals) {
        bool above_lower = interval.lower_infinite || x >= interval.lower;
        bool below_upper = interval.upper_infinite || x <= interval.upper;
        if (above_lower && below_upper) {
            return true;
        }
    }
    return false;
}

double SearchMetrics::initial_mask_gbs() const {
    if (sieve_ms == 0.0) {
        return 0.0;
    }
    return static_cast<double>(initial_mask_bytes / (sieve_ms * 1.0e6L));
}

struct PointSearch::Impl {
    explicit Impl(SearchOptions search_options)
        : options(validate_search_options(std::move(search_options))),
          polynomial(options.coefficients) {}

    SearchOptions options;
    ExactPolynomial polynomial;
};

PointSearch::PointSearch(SearchOptions options)
    : impl_(new Impl(std::move(options))) {}

PointSearch::~PointSearch() = default;

SearchMetrics PointSearch::run(const PointCallback &point_callback) {
    SearchMetrics metrics;
    mpz_class root;
    if (impl_->options.include_infinity
        && impl_->polynomial.square_root(1, 0, root)
        && !point_callback({1, root.get_str(), 0})) {
        return metrics;
    }

    int first = impl_->options.denominators.first;
    while (first <= impl_->options.denominators.last) {
        int remaining = impl_->options.denominators.last - first;
        int last = first + std::min(remaining, kDenominatorBatchSize - 1);
        DenominatorRange range{first, last};
        std::vector<PointPair> points;
        SieveResult sieve = run_modular_sieve(
            impl_->options.coefficients, impl_->options.numerator_bound, range,
            [&](const CandidateBatch &candidates) {
                append_verified_candidates(
                    candidates, impl_->polynomial, points);
            });

        metrics.basis_ms += sieve.metrics.basis_ms;
        metrics.sieve_ms += sieve.metrics.sieve_ms;
        metrics.word_count = sieve.metrics.word_count;
        metrics.denominator_count += static_cast<unsigned>(range.count());
        metrics.initial_mask_bytes += sieve.metrics.initial_mask_bytes;
        add_checked(metrics.modular_survivors, sieve.survivor_count,
                    "total survivor count overflow");

        sort_points(points);
        if (points.size() > std::numeric_limits<size_t>::max()
                                - metrics.exact_survivors) {
            throw std::overflow_error("exact survivor count overflow");
        }
        metrics.exact_survivors += points.size();

        for (const auto &point : points) {
            if (impl_->options.accepts(point.numerator, point.denominator)
                && !point_callback(point)) {
                return metrics;
            }
        }
        if (last == impl_->options.denominators.last) {
            break;
        }
        first = last + 1;
    }
    return metrics;
}

}  // namespace ratpoints_gpu
