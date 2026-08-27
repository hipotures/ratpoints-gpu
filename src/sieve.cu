#include "sieve.hpp"

#include <algorithm>
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

struct SurvivorState {
    unsigned long long count;
    bool paused;
};

struct SurvivorCounter {
    SurvivorCounter() : count(1), paused(1) {}

    void clear() {
        count.clear();
        paused.clear();
    }

    SurvivorState download_state() const {
        check_cuda(cudaDeviceSynchronize(), "sieve_kernel execution");
        return {count.download_scalar(), paused.download_scalar() != 0};
    }

    DeviceBuffer<unsigned long long> count;
    DeviceBuffer<unsigned> paused;
};

struct DeviceWorkspace {
    explicit DeviceWorkspace(const SievePlan &plan)
        : masks(plan.mask_count), affine(kMaximumPrime),
          inverses(kMaximumPrime),
          primes(plan.primes.size()), offsets(plan.mask_offsets.size()),
          selected(plan.selected_primes.size()),
          next_iterations(static_cast<size_t>(plan.denominator_range.count())
                          * kWarpsPerBlock) {
        primes.upload(plan.primes);
        offsets.upload(plan.mask_offsets);
        selected.upload(plan.selected_primes);
        next_iterations.clear();
    }

    DeviceBuffer<uint32_t> masks;
    DeviceBuffer<uint8_t> affine;
    DeviceBuffer<int> inverses;
    DeviceBuffer<int> primes;
    DeviceBuffer<long long> offsets;
    DeviceBuffer<int> selected;
    DeviceBuffer<unsigned long long> next_iterations;
    SurvivorCounter survivors;
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
        unsigned amount, unsigned long long *count, unsigned *paused,
        unsigned long long capacity) {
    if (atomicCAS(paused, 0U, 0U)) {
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
    atomicExch(paused, 1U);
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

__device__ __forceinline__ void emit_survivors(
        uint32_t surviving, unsigned long long index, long long word,
        long long bound, int denominator,
        long long *__restrict__ output_numerators,
        int *__restrict__ output_denominators) {
    while (surviving) {
        int bit = __ffs(surviving) - 1;
        surviving &= surviving - 1;
        long long numerator = -bound + 32LL * word + bit;
        output_numerators[index] = numerator;
        output_denominators[index++] = denominator;
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

template <typename Word, bool Resume>
__global__ void sieve_kernel(const uint32_t *__restrict__ masks,
                             const int *__restrict__ primes,
                             const long long *__restrict__ offsets,
                             const int *__restrict__ selected,
                             Word words, long long bound,
                             int min_denominator,
                             long long *__restrict__ output_numerators,
                             int *__restrict__ output_denominators,
                             unsigned long long *__restrict__ count,
                             unsigned *__restrict__ paused,
                             unsigned long long *__restrict__ next_iterations,
                             unsigned long long capacity) {
#ifdef SHARED_MASK_ROWS
    extern __shared__ uint32_t local_masks[];
    __shared__ int local_row_offsets[kInitialPrimeCount];
#endif
    __shared__ int local_primes[kPrimeCount];
    __shared__ int denominator_residues[kPrimeCount];
    __shared__ long long local_offsets[kPrimeCount];
    int lane = threadIdx.x % kWarpSize;
    int warp = threadIdx.x / kWarpSize;
    // Each warp owns one progress slot and advances one block-sized stride per
    // iteration. Completed slots naturally resume beyond the word range.
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
    unsigned long long iteration = Resume
        ? next_iterations[progress_index] : 0;
    unsigned long long warp_word = static_cast<unsigned long long>(warp)
        * kWarpSize + iteration * kBlockSize;
#pragma unroll
    for (int i = 0; i < kInitialPrimeCount; ++i) {
#ifdef SHARED_MASK_ROWS
        residues[i] = local_row_offsets[i] + (Resume
            ? (warp_word + lane) % local_primes[i] : threadIdx.x);
#else
        residues[i] = Resume
            ? (warp_word + lane) % local_primes[i] : threadIdx.x;
#endif
    }
    for (; warp_word < static_cast<unsigned long long>(words);
         warp_word += kBlockSize, ++iteration) {
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
        }
        if (word == words - 1) {
            int valid_bits = static_cast<int>(bound
                - (-bound + 32LL * static_cast<long long>(word)) + 1);
            if (valid_bits < 32) {
                surviving &= (1u << valid_bits) - 1;
            }
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
                warp_count, count, paused, capacity);
        }
        output_index = __shfl_sync(0xffffffffu, output_index, 0);
        if (output_index == ULLONG_MAX) {
            if (lane == 0) {
                next_iterations[progress_index] = iteration;
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
    if (lane == 0) {
        next_iterations[progress_index] = iteration;
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

void configure_sieve_kernel(const SievePlan &plan) {
#ifdef SHARED_MASK_ROWS
    check_cuda(cudaFuncSetAttribute(sieve_kernel<unsigned, false>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   plan.shared_mask_bytes), "cudaFuncSetAttribute");
    check_cuda(cudaFuncSetAttribute(sieve_kernel<unsigned, true>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   plan.shared_mask_bytes), "cudaFuncSetAttribute");
    check_cuda(cudaFuncSetAttribute(sieve_kernel<long long, false>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   plan.shared_mask_bytes), "cudaFuncSetAttribute");
    check_cuda(cudaFuncSetAttribute(sieve_kernel<long long, true>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   plan.shared_mask_bytes), "cudaFuncSetAttribute");
#else
    (void)plan;
#endif
}

template <typename Word, bool Resume>
void launch_sieve_kernel(const SievePlan &plan, DeviceWorkspace &workspace,
                         DeviceSurvivors &survivors,
                         unsigned long long capacity, Word words) {
#ifdef SHARED_MASK_ROWS
    sieve_kernel<Word, Resume><<<plan.denominator_range.count(), kBlockSize,
        plan.shared_mask_bytes>>>(
#else
    sieve_kernel<Word, Resume><<<plan.denominator_range.count(), kBlockSize>>>(
#endif
        workspace.masks.data(), workspace.primes.data(),
        workspace.offsets.data(), workspace.selected.data(), words,
        plan.numerator_bound, plan.denominator_range.first,
        survivors.numerators.data(), survivors.denominators.data(),
        workspace.survivors.count.data(), workspace.survivors.paused.data(),
        workspace.next_iterations.data(),
        capacity);
}

void launch_sieve(const SievePlan &plan, DeviceWorkspace &workspace,
                   DeviceSurvivors &survivors,
                   unsigned long long capacity, bool resume) {
    workspace.survivors.clear();
    if (plan.word_count
        <= static_cast<long long>(UINT32_MAX) - (kBlockSize - 1)) {
        if (resume) {
            launch_sieve_kernel<unsigned, true>(
                plan, workspace, survivors, capacity,
                static_cast<unsigned>(plan.word_count));
        } else {
            launch_sieve_kernel<unsigned, false>(
                plan, workspace, survivors, capacity,
                static_cast<unsigned>(plan.word_count));
        }
    } else {
        if (resume) {
            launch_sieve_kernel<long long, true>(
                plan, workspace, survivors, capacity, plan.word_count);
        } else {
            launch_sieve_kernel<long long, false>(
                plan, workspace, survivors, capacity, plan.word_count);
        }
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
    SievePlan plan(numerator_bound, denominators);
    DeviceWorkspace workspace(plan);
    configure_sieve_kernel(plan);

    EventInterval basis_timer;
    EventInterval sieve_timer;
    basis_timer.start();
    build_mask_basis(plan, workspace, coefficients);
    basis_timer.stop();

    size_t capacity = initial_survivor_capacity(plan);
    DeviceSurvivors survivors(capacity);
    float sieve_ms = 0.0f;
    unsigned long long survivor_count = 0;
    bool resume = false;
    while (true) {
        sieve_timer.start();
        launch_sieve(plan, workspace, survivors, capacity, resume);
        sieve_timer.stop();
        SurvivorState state = workspace.survivors.download_state();
        sieve_ms += sieve_timer.milliseconds();
        if (state.count > std::numeric_limits<unsigned long long>::max()
                              - survivor_count) {
            throw std::overflow_error("total survivor count overflow");
        }
        survivor_count += state.count;
        if (state.count != 0) {
            CandidateBatch candidates = download_survivors(
                survivors, static_cast<size_t>(state.count));
            callback(candidates);
        }
        if (!state.paused) {
            break;
        }
        resume = true;
    }

    SieveMetrics metrics;
    metrics.basis_ms = basis_timer.milliseconds();
    metrics.sieve_ms = sieve_ms;
    metrics.word_count = static_cast<unsigned long long>(plan.word_count);
    metrics.initial_mask_bytes =
        static_cast<long double>(denominators.count()) * plan.word_count
        * kInitialPrimeCount * sizeof(uint32_t);
    return {survivor_count, metrics};
}

}  // namespace ratpoints_gpu
