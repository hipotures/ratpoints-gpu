#ifndef RATPOINTS_GPU_COMMAND_LINE_HPP
#define RATPOINTS_GPU_COMMAND_LINE_HPP

#include <cstdio>

#include "output_formatter.hpp"
#include "point_search.hpp"

namespace ratpoints_gpu {

struct ProgramOptions {
    SearchOptions search;
    OutputOptions output;
    bool quiet = false;
    bool verbose = false;
    bool stop_after_first = false;
};

bool help_requested(int argc, char **argv);
void print_help(std::FILE *stream);
ProgramOptions parse_command_line(int argc, char **argv);

}  // namespace ratpoints_gpu

#endif
