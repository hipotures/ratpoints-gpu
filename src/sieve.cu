#include "sieve.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cuda_support.hpp"
#include "sieve_plan.hpp"

#ifndef GLOBAL_MASK_ROWS
#define SHARED_MASK_ROWS
#endif

namespace ratpoints_gpu {
namespace {

using sieve_detail::SievePlan;
using sieve_detail::build_affine_mask;
using sieve_detail::kBlockSize;
using sieve_detail::kInitialPrimeCount;
using sieve_detail::kMaximumPrime;
using sieve_detail::kPrimeCount;
using sieve_detail::mod_inverse;

static_assert(kBlockSize > 0 && kBlockSize <= 1024,
              "invalid CUDA block size");
static_assert(kBlockSize % 32 == 0,
              "CUDA block size must contain whole warps");
static_assert(kPrimeCount <= kBlockSize,
              "CUDA block must initialize every selected prime");

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kBlockSize / kWarpSize;
constexpr size_t kMinimumSurvivorCapacity = 4096;
constexpr size_t kMaximumSurvivorCapacity = 1 << 24;
constexpr size_t kSurvivorMemoryBudgetDivisor = 4;

struct OutputState {
    unsigned long long count;
    bool output_full;
};

struct DeviceOutputState {
    DeviceOutputState() : count(1), output_full(1) {}

    void clear() {
        count.clear();
        output_full.clear();
    }

    OutputState download() const {
        check_cuda(cudaDeviceSynchronize(), "sieve_kernel execution");
        return {count.download_scalar(), output_full.download_scalar() != 0};
    }

    DeviceBuffer<unsigned long long> count;
    DeviceBuffer<unsigned> output_full;
};

struct DeviceWorkspace {
    explicit DeviceWorkspace(const SievePlan &plan)
        : masks(plan.mask_count), affine(kMaximumPrime),
          inverses(kMaximumPrime),
          primes(plan.primes.size()), offsets(plan.mask_offsets.size()),
          selected(plan.selected_primes.size()),
          resume_words(static_cast<size_t>(plan.denominator_range.count())
                       * kWarpsPerBlock) {
        primes.upload(plan.primes);
        offsets.upload(plan.mask_offsets);
        selected.upload(plan.selected_primes);
        resume_words.clear();
    }

    DeviceBuffer<uint32_t> masks;
    DeviceBuffer<uint8_t> affine;
    DeviceBuffer<int> inverses;
    DeviceBuffer<int> primes;
    DeviceBuffer<long long> offsets;
    DeviceBuffer<int> selected;
    DeviceBuffer<unsigned long long> resume_words;
    DeviceOutputState output_state;
};

struct DeviceSurvivors {
    explicit DeviceSurvivors(size_t size)
        : numerators(size), denominators(size) {}

    DeviceBuffer<long long> numerators;
    DeviceBuffer<int> denominators;
};

size_t initial_survivor_capacity(const SievePlan &plan) {
    long double sites = static_cast<long double>(
        plan.denominator_range.count()) * (2.0L * plan.numerator_bound + 1.0L);
    // Approximate each fast-stage prime as halving the candidates. Dense cases
    // stream in additional launches without increasing the fixed allocation.
    long double estimate = std::ldexp(sites, -kInitialPrimeCount);
    estimate = std::max(estimate, static_cast<long double>(
        plan.denominator_range.count()));
    estimate = std::max(estimate,
                        static_cast<long double>(kMinimumSurvivorCapacity));
    estimate = std::min(estimate,
                        static_cast<long double>(kMaximumSurvivorCapacity));
    size_t capacity = static_cast<size_t>(std::ceil(estimate));
    size_t free_bytes;
    size_t total_bytes;
    check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    size_t memory_limit = free_bytes
        / (kSurvivorMemoryBudgetDivisor
           * (sizeof(long long) + sizeof(int)));
    return std::max(kMinimumSurvivorCapacity,
                    std::min(capacity, memory_limit));
}

// A warp commits all candidates from its current iteration or none of them,
// so a failed reservation can safely resume that iteration after a flush.
__device__ __forceinline__ unsigned long long reserve_survivors(
        unsigned amount, unsigned long long *count, unsigned *output_full,
        unsigned long long capacity) {
    if (atomicCAS(output_full, 0U, 0U)) {
        return ULLONG_MAX;
    }
    unsigned long long current = atomicCAS(count, 0ULL, 0ULL);
    while (amount <= capacity - current) {
        unsigned long long previous = atomicCAS(count, current,
                                                current + amount);
        if (previous == current) {
            return current;
        }
        current = previous;
    }
    atomicExch(output_full, 1U);
    return ULLONG_MAX;
}

__global__ void basis_kernel(uint32_t *__restrict__ masks,
                             const uint8_t *__restrict__ affine,
                             const int *__restrict__ inverses,
                             int prime, long long offset, long long bound) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= prime * prime) {
        return;
    }
    int denominator_residue = id / prime;
    int word_residue = id % prime;
    if (denominator_residue == 0) {
        masks[offset + id] = 0;
        return;
    }
    int inverse = inverses[denominator_residue];
    long long numerator = (-bound + 32LL * word_residue) % prime;
    if (numerator < 0) {
        numerator += prime;
    }
    int x = numerator * inverse % prime;
    uint32_t packed = 0;
    for (int bit = 0; bit < 32; ++bit) {
        if (affine[x]) {
            packed |= 1u << bit;
        }
        x += inverse;
        if (x >= prime) {
            x -= prime;
        }
    }
    masks[offset + id] = packed;
}

__device__ __forceinline__ void emit_survivor(
        unsigned long long index, long long word, int bit,
        long long bound, int denominator,
        long long *__restrict__ output_numerators,
        int *__restrict__ output_denominators) {
    output_numerators[index] = -bound + 32LL * word + bit;
    output_denominators[index] = denominator;
}

__device__ __forceinline__ void emit_survivors(
        uint32_t surviving, unsigned long long index, long long word,
        long long bound, int denominator,
        long long *__restrict__ output_numerators,
        int *__restrict__ output_denominators) {
    while (surviving) {
        int bit = __ffs(surviving) - 1;
        surviving &= surviving - 1;
        emit_survivor(index++, word, bit, bound, denominator,
                      output_numerators, output_denominators);
    }
}

template <typename Word>
__device__ __noinline__ uint32_t sieve_late_primes(
        uint32_t surviving, Word word,
        const uint32_t *__restrict__ masks,
        const int *local_primes, const int *denominator_residues,
        const long long *local_offsets) {
    for (int i = kInitialPrimeCount; i < kPrimeCount; ++i) {
        int prime = local_primes[i];
        unsigned residue = word % static_cast<Word>(prime);
        surviving &= masks[local_offsets[i]
            + static_cast<long long>(denominator_residues[i]) * prime
            + residue];
        if (!surviving) {
            break;
        }
    }
    return surviving;
}

enum class SieveMode {
    fast,
    streaming_initial,
    streaming_resume,
};

template <typename Word, SieveMode Mode>
__global__ void sieve_kernel(const uint32_t *__restrict__ masks,
                             const int *__restrict__ primes,
                             const long long *__restrict__ offsets,
                             const int *__restrict__ selected,
                             Word words, long long bound,
                             int min_denominator,
                             long long *__restrict__ output_numerators,
                             int *__restrict__ output_denominators,
                             unsigned long long *__restrict__ count,
                             unsigned *__restrict__ output_full,
                             unsigned long long *__restrict__ resume_words,
                             unsigned long long capacity) {
    constexpr bool streaming = Mode != SieveMode::fast;
    constexpr bool resume = Mode == SieveMode::streaming_resume;
#ifdef SHARED_MASK_ROWS
    extern __shared__ uint32_t local_masks[];
    __shared__ int local_row_offsets[kInitialPrimeCount];
#endif
    __shared__ int local_primes[kPrimeCount];
    __shared__ int denominator_residues[kPrimeCount];
    __shared__ long long local_offsets[kPrimeCount];
    int lane = threadIdx.x % kWarpSize;
    int warp = threadIdx.x / kWarpSize;
    // Each warp owns one slot containing its first uncommitted word. Completed
    // slots contain a word at or beyond the end of the range.
    size_t progress_index = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock
        + warp;
    int denominator = static_cast<int>(
        static_cast<long long>(blockIdx.x) + min_denominator);
    if (threadIdx.x < kPrimeCount) {
        int index = selected[blockIdx.x * kPrimeCount + threadIdx.x];
        int prime = primes[index];
        local_primes[threadIdx.x] = prime;
        denominator_residues[threadIdx.x] = denominator % prime;
        local_offsets[threadIdx.x] = offsets[index];
    }
    __syncthreads();
#ifdef SHARED_MASK_ROWS
    if (threadIdx.x == 0) {
        int row_offset = 0;
        for (int i = 0; i < kInitialPrimeCount; ++i) {
            local_row_offsets[i] = row_offset;
            row_offset += local_primes[i];
        }
    }
    __syncthreads();
    for (int i = 0; i < kInitialPrimeCount; ++i) {
        int prime = local_primes[i];
        for (int residue = threadIdx.x; residue < prime;
             residue += kBlockSize) {
            local_masks[local_row_offsets[i] + residue] =
                masks[local_offsets[i]
                    + static_cast<long long>(denominator_residues[i]) * prime
                    + residue];
        }
    }
    __syncthreads();
#endif
    unsigned residues[kInitialPrimeCount];
    Word warp_word = resume
        ? static_cast<Word>(resume_words[progress_index])
        : static_cast<Word>(warp * kWarpSize);
#pragma unroll
    for (int i = 0; i < kInitialPrimeCount; ++i) {
#ifdef SHARED_MASK_ROWS
        residues[i] = local_row_offsets[i] + (resume
            ? (warp_word + lane) % local_primes[i] : threadIdx.x);
#else
        residues[i] = resume
            ? (warp_word + lane) % local_primes[i] : threadIdx.x;
#endif
    }
    for (; warp_word < words; warp_word += kBlockSize) {
        Word word = static_cast<Word>(warp_word + lane);
        uint32_t surviving = 0xffffffffu;
        if (word >= words) {
            surviving = 0;
        }
#pragma unroll
        for (int i = 0; i < kInitialPrimeCount; ++i) {
            int prime = local_primes[i];
#ifdef SHARED_MASK_ROWS
            surviving &= local_masks[residues[i]];
#else
            surviving &= masks[local_offsets[i]
                + static_cast<long long>(denominator_residues[i]) * prime
                + residues[i]];
#endif
            unsigned next = residues[i] + kBlockSize;
#ifdef SHARED_MASK_ROWS
            unsigned row_end = local_row_offsets[i] + prime;
            residues[i] = next >= row_end ? next - prime : next;
#else
            residues[i] = next >= static_cast<unsigned>(prime)
                ? next - prime : next;
#endif
        }
        if (surviving) {
            surviving = sieve_late_primes(surviving, word, masks,
                local_primes, denominator_residues, local_offsets);
            if (surviving && word == words - 1) {
                int valid_bits = static_cast<int>(bound
                    - (-bound + 32LL * static_cast<long long>(word)) + 1);
                if (valid_bits < 32) {
                    surviving &= (1u << valid_bits) - 1;
                }
            }
        }
        if (!streaming) {
            // Keep the common case as lean as the fixed-buffer kernel. An
            // overflow marks this output incomplete, so the host discards it and
            // reruns through the resumable path below.
            while (surviving) {
                int bit = __ffs(surviving) - 1;
                surviving &= surviving - 1;
                unsigned long long output_index = atomicAdd(count, 1ULL);
                if (output_index < capacity) {
                    emit_survivor(output_index, static_cast<long long>(word),
                                  bit, bound, denominator, output_numerators,
                                  output_denominators);
                } else {
                    atomicExch(output_full, 1U);
                    return;
                }
            }
            continue;
        }
        unsigned local_count = __popc(surviving);
        if (!__any_sync(0xffffffffu, local_count != 0)) {
            continue;
        }
        unsigned inclusive = local_count;
#pragma unroll
        for (int offset = 1; offset < kWarpSize; offset *= 2) {
            unsigned previous = __shfl_up_sync(
                0xffffffffu, inclusive, offset);
            if (lane >= offset) {
                inclusive += previous;
            }
        }
        unsigned warp_count = __shfl_sync(0xffffffffu, inclusive,
                                          kWarpSize - 1);
        unsigned long long output_index = 0;
        if (lane == 0 && warp_count != 0) {
            output_index = reserve_survivors(
                warp_count, count, output_full, capacity);
        }
        output_index = __shfl_sync(0xffffffffu, output_index, 0);
        if (output_index == ULLONG_MAX) {
            if (lane == 0) {
                resume_words[progress_index] = warp_word;
            }
            return;
        }
        if (surviving) {
            emit_survivors(surviving,
                output_index + inclusive - local_count,
                static_cast<long long>(word), bound, denominator,
                output_numerators, output_denominators);
        }
    }
    if (streaming && lane == 0) {
        resume_words[progress_index] = warp_word;
    }
}

void build_mask_basis(const SievePlan &plan, DeviceWorkspace &workspace,
                      const Coefficients &coefficients) {
    for (size_t i = 0; i < plan.primes.size(); ++i) {
        int prime = plan.primes[i];
        std::vector<uint8_t> affine =
            build_affine_mask(prime, coefficients);
        std::vector<int> inverses(prime);
        for (int value = 1; value < prime; ++value) {
            inverses[value] = mod_inverse(value, prime);
        }
        workspace.affine.upload(affine);
        workspace.inverses.upload(inverses);
        basis_kernel<<<(prime * prime + kBlockSize - 1) / kBlockSize,
                       kBlockSize>>>(
            workspace.masks.data(), workspace.affine.data(),
            workspace.inverses.data(), prime, plan.mask_offsets[i],
            plan.numerator_bound);
        check_cuda(cudaGetLastError(), "basis_kernel launch");
    }
}

template <typename Word, SieveMode Mode>
void configure_sieve_specialization(const SievePlan &plan) {
#ifdef SHARED_MASK_ROWS
    check_cuda(cudaFuncSetAttribute(sieve_kernel<Word, Mode>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   plan.shared_mask_bytes),
               "cudaFuncSetAttribute");
#else
    (void)plan;
#endif
}

template <typename Word>
void configure_sieve_specializations(const SievePlan &plan) {
    configure_sieve_specialization<Word, SieveMode::fast>(plan);
    configure_sieve_specialization<Word, SieveMode::streaming_initial>(plan);
    configure_sieve_specialization<Word, SieveMode::streaming_resume>(plan);
}

void configure_sieve_kernel(const SievePlan &plan) {
    configure_sieve_specializations<unsigned>(plan);
    configure_sieve_specializations<long long>(plan);
}

template <typename Word, SieveMode Mode>
void launch_sieve_kernel(const SievePlan &plan, DeviceWorkspace &workspace,
                         DeviceSurvivors &survivors,
                         unsigned long long capacity, Word words) {
#ifdef SHARED_MASK_ROWS
    sieve_kernel<Word, Mode><<<plan.denominator_range.count(), kBlockSize,
                               plan.shared_mask_bytes>>>(
#else
    sieve_kernel<Word, Mode><<<plan.denominator_range.count(), kBlockSize>>>(
#endif
        workspace.masks.data(), workspace.primes.data(),
        workspace.offsets.data(), workspace.selected.data(), words,
        plan.numerator_bound, plan.denominator_range.first,
        survivors.numerators.data(), survivors.denominators.data(),
        workspace.output_state.count.data(),
        workspace.output_state.output_full.data(),
        workspace.resume_words.data(), capacity);
}

template <typename Word>
void launch_sieve_mode(const SievePlan &plan, DeviceWorkspace &workspace,
                       DeviceSurvivors &survivors,
                       unsigned long long capacity, Word words,
                       SieveMode mode) {
    switch (mode) {
    case SieveMode::fast:
        launch_sieve_kernel<Word, SieveMode::fast>(
            plan, workspace, survivors, capacity, words);
        break;
    case SieveMode::streaming_initial:
        launch_sieve_kernel<Word, SieveMode::streaming_initial>(
            plan, workspace, survivors, capacity, words);
        break;
    case SieveMode::streaming_resume:
        launch_sieve_kernel<Word, SieveMode::streaming_resume>(
            plan, workspace, survivors, capacity, words);
        break;
    }
}

void launch_sieve(const SievePlan &plan, DeviceWorkspace &workspace,
                  DeviceSurvivors &survivors,
                  unsigned long long capacity, SieveMode mode) {
    workspace.output_state.clear();
    if (plan.word_count
        <= static_cast<long long>(UINT32_MAX) - (kBlockSize - 1)) {
        launch_sieve_mode(plan, workspace, survivors, capacity,
                          static_cast<unsigned>(plan.word_count), mode);
    } else {
        launch_sieve_mode(plan, workspace, survivors, capacity,
                          plan.word_count, mode);
    }
    check_cuda(cudaGetLastError(), "sieve_kernel launch");
}

CandidateBatch download_survivors(const DeviceSurvivors &survivors,
                                  size_t count) {
    CandidateBatch candidates;
    candidates.numerators.resize(count);
    candidates.denominators.resize(count);
    survivors.numerators.download(candidates.numerators);
    survivors.denominators.download(candidates.denominators);
    return candidates;
}

}  // namespace

SieveResult run_modular_sieve(const Coefficients &coefficients,
                               long long numerator_bound,
                               DenominatorRange denominators,
                               const CandidateBatchCallback &callback) {
    using Clock = std::chrono::steady_clock;
    auto milliseconds = [](Clock::time_point first, Clock::time_point last) {
        return std::chrono::duration<double, std::milli>(last - first).count();
    };
    const auto plan_start = Clock::now();
    SievePlan plan(numerator_bound, denominators);
    const auto workspace_start = Clock::now();
    DeviceWorkspace workspace(plan);
    configure_sieve_kernel(plan);
    const auto basis_start = Clock::now();

    EventInterval basis_timer;
    EventInterval sieve_timer;
    basis_timer.start();
    build_mask_basis(plan, workspace, coefficients);
    basis_timer.stop();
    const auto survivor_setup_start = Clock::now();

    size_t capacity = initial_survivor_capacity(plan);
    DeviceSurvivors survivors(capacity);
    const auto sieve_start = Clock::now();
    float sieve_ms = 0.0f;
    double verification_ms = 0.0;
    unsigned long long survivor_count = 0;
    SieveMode mode = SieveMode::fast;
    while (true) {
        sieve_timer.start();
        launch_sieve(plan, workspace, survivors, capacity, mode);
        sieve_timer.stop();
        OutputState state = workspace.output_state.download();
        sieve_ms += sieve_timer.milliseconds();
        if (mode == SieveMode::fast && state.output_full) {
            // No fast-path output is consumed until its capacity check passes.
            workspace.resume_words.clear();
            mode = SieveMode::streaming_initial;
            continue;
        }
        if (state.count > capacity) {
            throw std::logic_error("streaming survivor count exceeds capacity");
        }
        if (state.count > std::numeric_limits<unsigned long long>::max()
                              - survivor_count) {
            throw std::overflow_error("total survivor count overflow");
        }
        survivor_count += state.count;
        if (state.count != 0) {
            CandidateBatch candidates = download_survivors(
                survivors, static_cast<size_t>(state.count));
            const auto verification_start = Clock::now();
            callback(candidates);
            verification_ms += milliseconds(verification_start, Clock::now());
        }
        if (!state.output_full) {
            break;
        }
        mode = SieveMode::streaming_resume;
    }

    SieveMetrics metrics;
    metrics.basis_ms = basis_timer.milliseconds();
    metrics.sieve_ms = sieve_ms;
    metrics.plan_ms = milliseconds(plan_start, workspace_start);
    metrics.workspace_ms = milliseconds(workspace_start, basis_start);
    metrics.basis_wall_ms = milliseconds(basis_start, survivor_setup_start);
    metrics.survivor_setup_ms = milliseconds(survivor_setup_start, sieve_start);
    metrics.verification_ms = verification_ms;
    metrics.sieve_wall_ms = milliseconds(sieve_start, Clock::now()) - verification_ms;
    metrics.word_count = static_cast<unsigned long long>(plan.word_count);
    metrics.initial_mask_bytes =
        static_cast<long double>(denominators.count()) * plan.word_count
        * kInitialPrimeCount * sizeof(uint32_t);
    return {survivor_count, metrics};
}

}  // namespace ratpoints_gpu
