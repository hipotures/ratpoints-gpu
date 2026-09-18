#include "command_line.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ratpoints_gpu {
namespace {

constexpr int kMaximumDegree = 100;
constexpr long long kMaximumHeight =
    (std::numeric_limits<long long>::max() - 32) / 2;
constexpr char kUsage[] =
    "Usage: ratpoints_gpu 'c0 c1 ... cn' HEIGHT [OPTIONS]";

long long parse_positive(const char *text, const char *name) {
    errno = 0;
    char *end = nullptr;
    long long value = std::strtoll(text, &end, 10);
    if (errno == ERANGE || end == text || *end || value < 1) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return value;
}

int parse_positive_int(const char *text, const char *name) {
    long long value = parse_positive(text, name);
    if (value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string(name) + " exceeds GPU limit");
    }
    return static_cast<int>(value);
}

mpq_class parse_real(const char *text, const char *name) {
    std::string input(text);
    size_t index = 0;
    bool negative = false;
    if (index < input.size()
        && (input[index] == '+' || input[index] == '-')) {
        negative = input[index++] == '-';
    }

    std::string digits;
    size_t fractional_digits = 0;
    bool saw_digit = false;
    while (index < input.size() && input[index] >= '0' && input[index] <= '9') {
        digits.push_back(input[index++]);
        saw_digit = true;
    }
    if (index < input.size() && input[index] == '.') {
        ++index;
        size_t fraction_start = digits.size();
        while (index < input.size()
               && input[index] >= '0' && input[index] <= '9') {
            digits.push_back(input[index++]);
            saw_digit = true;
        }
        fractional_digits = digits.size() - fraction_start;
    }

    long long exponent = 0;
    bool exponent_negative = false;
    if (index < input.size() && (input[index] == 'e' || input[index] == 'E')) {
        ++index;
        if (index < input.size()
            && (input[index] == '+' || input[index] == '-')) {
            exponent_negative = input[index++] == '-';
        }
        size_t exponent_start = index;
        while (index < input.size()
               && input[index] >= '0' && input[index] <= '9') {
            int digit = input[index++] - '0';
            if (exponent > (10000 - digit) / 10) {
                throw std::invalid_argument(
                    std::string(name) + " exponent is too large");
            }
            exponent = exponent * 10 + digit;
        }
        if (index == exponent_start) {
            throw std::invalid_argument(std::string(name) + " must be real");
        }
    }
    if (!saw_digit || index != input.size()) {
        throw std::invalid_argument(std::string(name) + " must be real");
    }
    if (exponent_negative) {
        exponent = -exponent;
    }

    mpz_class numerator(digits, 10);
    if (negative) {
        numerator = -numerator;
    }
    long long scale = static_cast<long long>(fractional_digits) - exponent;
    mpz_class power;
    mpz_ui_pow_ui(power.get_mpz_t(), 10,
                  static_cast<unsigned long>(scale < 0 ? -scale : scale));
    mpq_class value = scale >= 0
        ? mpq_class(numerator, power)
        : mpq_class(numerator * power);
    value.canonicalize();
    return value;
}

bool is_integer(const std::string &text) {
    size_t index = !text.empty() && (text[0] == '-' || text[0] == '+');
    if (index == text.size()) {
        return false;
    }
    for (; index < text.size(); ++index) {
        if (text[index] < '0' || text[index] > '9') {
            return false;
        }
    }
    return true;
}

bool is_zero(const std::string &text) {
    size_t index = !text.empty() && (text[0] == '-' || text[0] == '+');
    for (; index < text.size(); ++index) {
        if (text[index] != '0') {
            return false;
        }
    }
    return true;
}

Coefficients split_coefficients(const char *text) {
    std::istringstream stream(text);
    Coefficients coefficients;
    for (std::string token; stream >> token;) {
        if (!token.empty() && token.front() == '+') {
            token.erase(0, 1);
        }
        coefficients.push_back(std::move(token));
    }
    return coefficients;
}

void validate_coefficients(const Coefficients &coefficients) {
    if (coefficients.size() < 2
        || coefficients.size() > static_cast<size_t>(kMaximumDegree + 1)) {
        throw std::invalid_argument(
            "supported polynomial degrees are 1 through 100");
    }
    for (const auto &coefficient : coefficients) {
        if (!is_integer(coefficient)) {
            throw std::invalid_argument("coefficients must be decimal integers");
        }
    }
    if (is_zero(coefficients.back())) {
        throw std::invalid_argument("leading coefficient must be nonzero");
    }
}

class OptionParser {
public:
    OptionParser(int argc, char **argv, ProgramOptions &result)
        : argc_(argc), argv_(argv), result_(result) {}

    void parse() {
        while (index_ < argc_) {
            std::string option(argv_[index_++]);
            if (parse_search_option(option)
                || parse_output_option(option)
                || parse_runtime_option(option)) {
                continue;
            }
            throw std::invalid_argument(
                "unsupported ratpoints option: " + option);
        }
        if (has_pending_lower_) {
            result_.search.intervals.push_back(
                {pending_lower_, mpq_class(), false, true});
        }
    }

private:
    const char *argument(const std::string &option) {
        if (index_ >= argc_) {
            throw std::invalid_argument(option + " requires an argument");
        }
        return argv_[index_++];
    }

    bool parse_search_option(const std::string &option) {
        if (option == "-dl") {
            result_.search.denominators.first = parse_positive_int(
                argument(option), "lower denominator");
        } else if (option == "-du") {
            result_.search.denominators.last = parse_positive_int(
                argument(option), "upper denominator");
        } else if (option == "-l") {
            begin_interval(option);
        } else if (option == "-u") {
            end_interval(argument(option));
        } else if (option == "-i") {
            result_.search.include_infinity = false;
        } else if (option == "-I") {
            result_.search.include_infinity = true;
        } else {
            return false;
        }
        return true;
    }

    bool parse_output_option(const std::string &option) {
        if (option == "-y") {
            result_.output.print_y = false;
        } else if (option == "-Y") {
            result_.output.print_y = true;
        } else if (option == "-z") {
            result_.output.print_points = false;
        } else if (option == "-Z") {
            result_.output.print_points = true;
        } else if (option == "-f") {
            result_.output.format = argument(option);
        } else if (option == "-fs") {
            result_.output.before = argument(option);
        } else if (option == "-fm") {
            result_.output.between = argument(option);
        } else if (option == "-fe") {
            result_.output.after = argument(option);
        } else {
            return false;
        }
        return true;
    }

    bool parse_runtime_option(const std::string &option) {
        if (option == "--devices") {
            result_.search.devices = parse_device_selection(argument(option));
        } else if (option == "--batch-size") {
            result_.search.denominator_batch_size = parse_positive_int(
                argument(option), "denominator batch size");
            if (result_.search.denominator_batch_size > (1 << 16)) {
                throw std::invalid_argument("--batch-size must not exceed 65536");
            }
        } else if (option == "--square-denominators") {
            result_.search.square_denominators = true;
        } else if (option == "-q") {
            result_.quiet = true;
        } else if (option == "-v") {
            result_.verbose = true;
        } else if (option == "-1") {
            result_.stop_after_first = true;
        } else if (option == "-s" || option == "-k" || option == "-K"
                   || option == "-j" || option == "-J") {
            // CPU-only optimizations do not change the searched set.
        } else {
            return false;
        }
        return true;
    }

    void begin_interval(const std::string &option) {
        if (has_pending_lower_) {
            throw std::invalid_argument(
                "interval lower bound without upper bound");
        }
        pending_lower_ = parse_real(
            argument(option), "interval lower bound");
        has_pending_lower_ = true;
    }

    void end_interval(const char *text) {
        mpq_class upper = parse_real(text, "interval upper bound");
        if (has_pending_lower_ && pending_lower_ > upper) {
            throw std::invalid_argument(
                "interval lower bound exceeds upper bound");
        }
        if (!result_.search.intervals.empty()
            && (!has_pending_lower_
                || result_.search.intervals.back().upper_infinite
                || pending_lower_ < result_.search.intervals.back().upper)) {
            throw std::invalid_argument("search intervals must be ordered");
        }
        result_.search.intervals.push_back(
            {pending_lower_, upper, !has_pending_lower_, false});
        has_pending_lower_ = false;
    }

    int argc_;
    char **argv_;
    ProgramOptions &result_;
    int index_ = 3;
    mpq_class pending_lower_;
    bool has_pending_lower_ = false;
};

}  // namespace

bool help_requested(int argc, char **argv) {
    return argc == 2 && std::string(argv[1]) == "--help";
}

void print_help(std::FILE *stream) {
    constexpr char help_text[] =
        "Usage: ratpoints_gpu 'c0 c1 ... cn' HEIGHT [OPTIONS]\n"
        "\n"
        "GPU selection (IDs are relative to CUDA_VISIBLE_DEVICES):\n"
        "  --devices all|0,1,...  select GPUs (default: all visible)\n"
        "  --batch-size N        denominators per GPU batch, 1..65536\n"
        "  --square-denominators  interpret -dl/-du as k bounds; search d=k^2\n"
        "  --list-devices        list GPUs and exit; use without a curve\n"
        "\n"
        "Supported options:\n"
        "  -dl B       set the lower denominator bound\n"
        "  -du B       set the upper denominator bound\n"
        "  -l L        begin a closed x interval\n"
        "  -u U        end a closed x interval\n"
        "  -1          stop after the first point\n"
        "  -i, -I      suppress or restore points at infinity\n"
        "  -y, -Y      suppress or restore ordinates\n"
        "  -z, -Z      suppress or restore point output\n"
        "  -q          suppress timing output\n"
        "  -v          print timing and survivor metrics\n"
        "  -f FORMAT   set the point format\n"
        "  -fs TEXT    write TEXT before point output\n"
        "  -fm TEXT    write TEXT between points\n"
        "  -fe TEXT    write TEXT after point output\n"
        "  -s          accepted with no effect\n"
        "  -k, -K      accepted with no effect\n"
        "  -j, -J      accepted with no effect\n"
        "  --help      show this help\n";
    if (std::fputs(help_text, stream) == EOF) {
        throw std::runtime_error("failed to write help output");
    }
}

ProgramOptions parse_command_line(int argc, char **argv) {
    if (argc < 3) {
        throw std::invalid_argument(kUsage);
    }

    ProgramOptions result;
    result.search.coefficients = split_coefficients(argv[1]);
    result.search.numerator_bound = parse_positive(argv[2], "maximum height");
    if (result.search.numerator_bound > kMaximumHeight) {
        throw std::invalid_argument("maximum height exceeds safe GPU limit");
    }
    result.search.denominators.last =
        result.search.numerator_bound > std::numeric_limits<int>::max()
            ? std::numeric_limits<int>::max()
            : static_cast<int>(result.search.numerator_bound);

    OptionParser(argc, argv, result).parse();
    if (result.search.denominators.last > result.search.numerator_bound) {
        result.search.denominators.last =
            static_cast<int>(result.search.numerator_bound);
    }
    if (result.search.square_denominators
        && result.search.denominators.last > 46340) {
        result.search.denominators.last = 46340;
    }
    if (result.search.denominators.first > result.search.denominators.last) {
        throw std::invalid_argument("empty denominator range");
    }
    validate_coefficients(result.search.coefficients);
    return result;
}

}  // namespace ratpoints_gpu
