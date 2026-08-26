#include "output_formatter.hpp"

#include <stdexcept>

namespace ratpoints_gpu {

OutputFormatter::OutputFormatter(const OutputOptions &options)
    : print_points_(options.print_points),
      print_y_(options.print_y),
      format_(decode(options.format.empty()
          ? (options.print_y ? "(%x : %y : %z)\\n" : "(%x : %z)\\n")
          : options.format)),
      before_(decode(options.before)),
      between_(decode(options.between)),
      after_(decode(options.after)) {
    if (print_points_) {
        write(before_);
    }
}

void OutputFormatter::point_pair(const PointPair &point,
                                 bool include_conjugate) {
    write_point(point.numerator, point.ordinate, point.denominator);
    if (include_conjugate && print_y_ && point.ordinate != "0") {
        write_point(point.numerator, "-" + point.ordinate, point.denominator);
    }
}

void OutputFormatter::write_point(long long numerator,
                                  const std::string &ordinate,
                                  int denominator) {
    if (!print_points_) {
        return;
    }
    if (emitted_) {
        write(between_);
    }
    std::string output;
    output.reserve(format_.size() + ordinate.size() + 32);
    for (size_t i = 0; i < format_.size(); ++i) {
        if (format_[i] != '%' || i + 1 == format_.size()) {
            output.push_back(format_[i]);
            continue;
        }
        char marker = format_[i + 1];
        if (marker == 'x') {
            output += std::to_string(numerator);
        } else if (marker == 'y' && print_y_) {
            output += ordinate;
        } else if (marker == 'z') {
            output += std::to_string(denominator);
        } else {
            output.push_back('%');
            output.push_back(marker);
        }
        ++i;
    }
    for (char &character : output) {
        if (character == kEscapedPercent) {
            character = '%';
        }
    }
    write(output);
    emitted_ = true;
}

void OutputFormatter::finish() {
    if (finished_) {
        return;
    }
    write(after_);
    if (std::fflush(stdout) == EOF) {
        throw std::runtime_error("failed to flush standard output");
    }
    finished_ = true;
}

std::string OutputFormatter::decode(const std::string &input) {
    std::string output;
    output.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] != '\\' || i + 1 == input.size()) {
            output.push_back(input[i]);
            continue;
        }
        char escaped = input[++i];
        if (escaped == 'n') {
            output.push_back('\n');
        } else if (escaped == 't') {
            output.push_back('\t');
        } else if (escaped == '%') {
            output.push_back(kEscapedPercent);
        } else {
            output.push_back('\\');
            output.push_back(escaped);
        }
    }
    return output;
}

void OutputFormatter::write(const std::string &text) {
    if (std::fputs(text.c_str(), stdout) == EOF) {
        throw std::runtime_error("failed to write standard output");
    }
}

}  // namespace ratpoints_gpu
