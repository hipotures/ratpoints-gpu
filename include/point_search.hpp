#ifndef RATPOINTS_GPU_POINT_SEARCH_HPP
#define RATPOINTS_GPU_POINT_SEARCH_HPP

#include <cstddef>
#include <functional>
#include <gmpxx.h>
#include <memory>
#include <vector>

#include "search_types.hpp"
#include "gpu_devices.hpp"

namespace ratpoints_gpu {

struct SearchInterval {
    mpq_class lower;
    mpq_class upper;
    bool lower_infinite = false;
    bool upper_infinite = false;
};

struct SearchOptions {
    Coefficients coefficients;
    long long numerator_bound = 0;
    DenominatorRange denominators{1, 0};
    std::vector<SearchInterval> intervals;
    bool include_infinity = true;
    std::vector<int> devices;  // Empty means all visible GPUs.
    int denominator_batch_size = 1 << 16;
    bool square_denominators = false;  // -dl/-du bound k, actual denominator k^2.

    bool accepts(long long numerator, int denominator) const;
};

struct DeviceSearchMetrics {
    GpuDevice device;
    unsigned long long denominator_count = 0;
    unsigned long long batches = 0;
    double basis_ms = 0.0;
    double sieve_ms = 0.0;
};

struct SearchMetrics {
    double wall_ms = 0.0;
    std::vector<DeviceSearchMetrics> devices;
    double basis_ms = 0.0;
    double sieve_ms = 0.0;
    double plan_ms = 0.0;
    double workspace_ms = 0.0;
    double basis_wall_ms = 0.0;
    double survivor_setup_ms = 0.0;
    double verification_ms = 0.0;
    double sieve_wall_ms = 0.0;
    unsigned long long word_count = 0;
    unsigned long long denominator_count = 0;
    unsigned long long modular_survivors = 0;
    long double initial_mask_bytes = 0.0L;
    size_t exact_survivors = 0;

    double initial_mask_gbs() const;
};

using PointCallback = std::function<bool(const PointPair &)>;

class PointSearch {
public:
    explicit PointSearch(SearchOptions options);
    ~PointSearch();

    PointSearch(const PointSearch &) = delete;
    PointSearch &operator=(const PointSearch &) = delete;

    SearchMetrics run(const PointCallback &point_callback);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ratpoints_gpu

#endif
