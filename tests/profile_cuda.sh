#!/usr/bin/env bash
# Profile a height-10M Stoll search. Usage: profile_cuda.sh nsys|ncu [binary] [output-prefix]
set -euo pipefail

mode=${1:?expected nsys or ncu}
binary=${2:-./ratpoints_gpu}
output=${3:-cuda-profile}
curve='247747600 -985905640 567207969 2396040466 52485681 -470135160 82342800'

case "$mode" in
    nsys)
        nsys profile --trace=cuda,nvtx,osrt --stats=true --force-overwrite=true \
            --output="$output" "$binary" "$curve" 10000000 \
            -dl 1 -du 10000000 --devices 0 -v >"$output.stdout"
        ;;
    ncu)
        # One full-size denominator batch keeps replay cost bounded.
        ncu --set full --target-processes all --kernel-name 'regex:sieve_kernel' \
            --launch-count 1 --force-overwrite --export="$output" \
            "$binary" "$curve" 10000000 -dl 1 -du 65536 \
            --devices 0 -v >"$output.stdout"
        ;;
    *)
        echo "expected nsys or ncu" >&2
        exit 2
        ;;
esac
