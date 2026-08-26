#include <cstdio>
#include <cstdlib>
#include <exception>

#include "command_line.hpp"
#include "output_formatter.hpp"
#include "point_search.hpp"

namespace ratpoints_gpu {
namespace {

void print_metrics(const SearchMetrics &metrics) {
    if (metrics.denominator_count == 0) {
        return;
    }
    std::fprintf(stderr,
                 "basis_ms=%.3f sieve_ms=%.3f words=%llu "
                 "initial_mask_gbs=%.3f survivors=%llu exact_survivors=%zu\n",
                 metrics.basis_ms, metrics.sieve_ms, metrics.word_count,
                 metrics.initial_mask_gbs(), metrics.modular_survivors,
                 metrics.exact_survivors);
}

}  // namespace
}  // namespace ratpoints_gpu

int main(int argc, char **argv) {
    try {
        if (argc == 1) {
            ratpoints_gpu::print_help(stderr);
            return 1;
        }
        if (ratpoints_gpu::help_requested(argc, argv)) {
            ratpoints_gpu::print_help(stdout);
            return 0;
        }
        ratpoints_gpu::ProgramOptions options =
            ratpoints_gpu::parse_command_line(argc, argv);
        ratpoints_gpu::PointSearch search(options.search);
        ratpoints_gpu::OutputFormatter output(options.output);
        ratpoints_gpu::SearchMetrics metrics = search.run(
            [&](const ratpoints_gpu::PointPair &point) {
                output.point_pair(point, !options.stop_after_first);
                return !options.stop_after_first;
            });
        output.finish();

        bool metrics_enabled = std::getenv("RATPOINTS_GPU_BENCHMARK") != nullptr
            || (options.verbose && !options.quiet);
        if (metrics_enabled) {
            ratpoints_gpu::print_metrics(metrics);
        }
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
