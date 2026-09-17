#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "command_line.hpp"
#include "gpu_devices.hpp"
#include "output_formatter.hpp"
#include "point_search.hpp"

namespace ratpoints_gpu {
namespace {

void print_device(std::FILE *stream, const GpuDevice &device) {
    std::fprintf(stream,
        "device=%d name=\"%s\" compute=%d.%d sm=%d memory_mib=%llu pci=%s\n",
        device.id, device.name.c_str(), device.major, device.minor,
        device.multiprocessors, device.memory_bytes / (1024ULL * 1024ULL),
        device.pci_bus_id.c_str());
}

void print_metrics(const SearchMetrics &metrics) {
    // Event times are sums over devices, NOT multi-GPU wall-clock times.
    std::fprintf(stderr,
        "wall_ms=%.3f devices=%zu denominators=%llu "
        "basis_ms=%.3f sieve_ms=%.3f words=%llu "
        "initial_mask_gbs=%.3f survivors=%llu exact_survivors=%zu\n",
        metrics.wall_ms, metrics.devices.size(), metrics.denominator_count,
        metrics.basis_ms, metrics.sieve_ms, metrics.word_count,
        metrics.initial_mask_gbs(), metrics.modular_survivors,
        metrics.exact_survivors);
    for (const auto &entry : metrics.devices) {
        print_device(stderr, entry.device);
        std::fprintf(stderr,
            "gpu=%d denominators=%llu batches=%llu basis_ms=%.3f sieve_ms=%.3f\n",
            entry.device.id, entry.denominator_count, entry.batches,
            entry.basis_ms, entry.sieve_ms);
    }
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
        if (argc == 2 && std::string(argv[1]) == "--list-devices") {
            for (const auto &device : ratpoints_gpu::available_gpu_devices()) {
                ratpoints_gpu::print_device(stdout, device);
            }
            return std::fflush(stdout) == EOF ? 1 : 0;
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
