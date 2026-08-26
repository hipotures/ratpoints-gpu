#ifndef RATPOINTS_GPU_OUTPUT_FORMATTER_HPP
#define RATPOINTS_GPU_OUTPUT_FORMATTER_HPP

#include <string>

#include "search_types.hpp"

namespace ratpoints_gpu {

struct OutputOptions {
    bool print_points = true;
    bool print_y = true;
    std::string format;
    std::string before;
    std::string between;
    std::string after;
};

class OutputFormatter {
public:
    explicit OutputFormatter(const OutputOptions &options);

    OutputFormatter(const OutputFormatter &) = delete;
    OutputFormatter &operator=(const OutputFormatter &) = delete;

    void point_pair(const PointPair &point, bool include_conjugate = true);
    void finish();

private:
    static constexpr char kEscapedPercent = '\x01';

    static std::string decode(const std::string &input);
    static void write(const std::string &text);
    void write_point(long long numerator, const std::string &ordinate,
                     int denominator);

    bool print_points_;
    bool print_y_;
    std::string format_;
    std::string before_;
    std::string between_;
    std::string after_;
    bool emitted_ = false;
    bool finished_ = false;
};

}  // namespace ratpoints_gpu

#endif
