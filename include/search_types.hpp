#ifndef RATPOINTS_GPU_SEARCH_TYPES_HPP
#define RATPOINTS_GPU_SEARCH_TYPES_HPP

#include <string>
#include <vector>

namespace ratpoints_gpu {

using Coefficients = std::vector<std::string>;

struct DenominatorRange {
    int first;
    int last;

    int count() const { return last - first + 1; }
};

struct ModularCandidate {
    long long numerator;
    int denominator;
};

struct PointPair {
    long long numerator;
    std::string ordinate;
    int denominator;
};

}  // namespace ratpoints_gpu

#endif
