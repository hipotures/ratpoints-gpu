#include "point_search.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "exact_polynomial.hpp"
#include "sieve.hpp"

namespace ratpoints_gpu {
namespace {

SearchOptions validate_search_options(SearchOptions options) {
    if (options.denominator_batch_size < 1
        || options.denominator_batch_size > (1 << 16)) {
        throw std::invalid_argument("denominator batch size must be in 1..65536");
    }
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
            points.push_back({numerator, root.get_str(), denominator});
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

struct VerifiedBatch {
    DenominatorRange range;
    SieveResult sieve;
    std::vector<PointPair> points;
};

VerifiedBatch search_batch(const SearchOptions &options, int device,
                            DenominatorRange range) {
    try {
        // Bind before any CUDA allocation, event, or kernel launch. All CUDA
        // resources are destroyed on this same thread before it returns.
        activate_gpu(device);
        ExactPolynomial polynomial(options.coefficients);
        VerifiedBatch batch;
        batch.range = range;
        batch.sieve = run_modular_sieve(
            options.coefficients, options.numerator_bound, range,
            [&](const CandidateBatch &candidates) {
                append_verified_candidates(candidates, polynomial, batch.points);
            });
        sort_points(batch.points);
        return batch;
    } catch (const std::exception &error) {
        throw std::runtime_error("GPU " + std::to_string(device)
            + ", denominators " + std::to_string(range.first) + ".."
            + std::to_string(range.last) + ": " + error.what());
    }
}

void accumulate(SearchMetrics &metrics, size_t device_index,
                 const VerifiedBatch &batch) {
    const auto &sieve = batch.sieve;
    auto &device = metrics.devices[device_index];
    device.basis_ms += sieve.metrics.basis_ms;
    device.sieve_ms += sieve.metrics.sieve_ms;
    device.denominator_count += static_cast<unsigned>(batch.range.count());
    ++device.batches;
    metrics.basis_ms += sieve.metrics.basis_ms;
    metrics.sieve_ms += sieve.metrics.sieve_ms;
    metrics.plan_ms += sieve.metrics.plan_ms;
    metrics.workspace_ms += sieve.metrics.workspace_ms;
    metrics.basis_wall_ms += sieve.metrics.basis_wall_ms;
    metrics.survivor_setup_ms += sieve.metrics.survivor_setup_ms;
    metrics.verification_ms += sieve.metrics.verification_ms;
    metrics.sieve_wall_ms += sieve.metrics.sieve_wall_ms;
    metrics.word_count = sieve.metrics.word_count;
    metrics.denominator_count += static_cast<unsigned>(batch.range.count());
    metrics.initial_mask_bytes += sieve.metrics.initial_mask_bytes;
    add_checked(metrics.modular_survivors, sieve.survivor_count,
                "total survivor count overflow");
    if (batch.points.size() > std::numeric_limits<size_t>::max()
                                  - metrics.exact_survivors) {
        throw std::overflow_error("exact survivor count overflow");
    }
    metrics.exact_survivors += batch.points.size();
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
          polynomial(options.coefficients),
          devices(select_gpu_devices(options.devices)) {}

    SearchOptions options;
    ExactPolynomial polynomial;
    std::vector<GpuDevice> devices;
};

PointSearch::PointSearch(SearchOptions options)
    : impl_(new Impl(std::move(options))) {}

PointSearch::~PointSearch() = default;

SearchMetrics PointSearch::run(const PointCallback &point_callback) {
    const auto started = std::chrono::steady_clock::now();
    SearchMetrics metrics;
    for (const auto &device : impl_->devices) {
        DeviceSearchMetrics entry;
        entry.device = device;
        metrics.devices.push_back(std::move(entry));
    }
    auto finish = [&]() {
        metrics.wall_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return metrics;
    };
    mpz_class root;
    if (impl_->options.include_infinity
        && impl_->polynomial.square_root(1, 0, root)
        && !point_callback({1, root.get_str(), 0})) {
        return finish();
    }

    // Process bounded waves in denominator order. There is no unbounded
    // reorder queue, no cross-device memory access, and no output from workers.
    long long first = impl_->options.denominators.first;
    const long long last = impl_->options.denominators.last;
    while (first <= last) {
        long long remaining = last - first + 1;
        size_t workers = static_cast<size_t>(std::min<long long>(
            remaining, static_cast<long long>(impl_->devices.size())));
        long long wave_size = std::min<long long>(remaining,
            static_cast<long long>(workers) * impl_->options.denominator_batch_size);
        std::vector<std::future<VerifiedBatch>> futures;
        std::vector<VerifiedBatch> batches;
        futures.reserve(workers);
        batches.reserve(workers);
        for (size_t i = 0; i < workers; ++i) {
            long long count = wave_size / static_cast<long long>(workers)
                + (static_cast<long long>(i) < wave_size % static_cast<long long>(workers));
            DenominatorRange range{static_cast<int>(first),
                                    static_cast<int>(first + count - 1)};
            first += count;  // 64-bit cursor also handles an INT_MAX endpoint.
            int device = impl_->devices[i].id;
            if (workers == 1) {
                batches.push_back(search_batch(impl_->options, device, range));
            } else {
                futures.push_back(std::async(std::launch::async,
                    [this, device, range]() {
                        return search_batch(impl_->options, device, range);
                    }));
            }
        }
        // Getting every result before output propagates errors from this wave.
        // If get(), allocation, or thread creation throws, async future
        // destruction joins all outstanding workers before options can die.
        for (auto &future : futures) {
            batches.push_back(future.get());
        }
        for (size_t i = 0; i < batches.size(); ++i) {
            accumulate(metrics, i, batches[i]);
        }
        for (const auto &batch : batches) {
            for (const auto &point : batch.points) {
                if (impl_->options.accepts(point.numerator, point.denominator)
                    && !point_callback(point)) {
                    return finish();
                }
            }
        }
    }
    return finish();
}

}  // namespace ratpoints_gpu
