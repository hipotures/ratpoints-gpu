#include "sieve.hpp"

#include <climits>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
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
static_assert(kPrimeCount <= kBlockSize,
              "CUDA block must initialize every selected prime");

struct SurvivorCounter {
    SurvivorCounter() : count(1), overflow(1) {}

    void clear() {
        count.clear();
        overflow.clear();
    }

    unsigned long long download_count() const {
        check_cuda(cudaDeviceSynchronize(), "sieve_kernel execution");
        if (overflow.download_scalar()) {
            throw std::overflow_error(
                "survivor count exceeds GPU counter range");
        }
        return count.download_scalar();
    }

    DeviceBuffer<unsigned long long> count;
    DeviceBuffer<unsigned> overflow;
};

struct DeviceWorkspace {
    explicit DeviceWorkspace(const SievePlan &plan)
        : masks(plan.mask_count), affine(kMaximumPrime),
          inverses(kMaximumPrime),
          primes(plan.primes.size()), offsets(plan.mask_offsets.size()),
          selected(plan.selected_primes.size()) {
        primes.upload(plan.primes);
        offsets.upload(plan.mask_offsets);
        selected.upload(plan.selected_primes);
    }

    DeviceBuffer<uint32_t> masks;
    DeviceBuffer<uint8_t> affine;
    DeviceBuffer<int> inverses;
    DeviceBuffer<int> primes;
    DeviceBuffer<long long> offsets;
    DeviceBuffer<int> selected;
    SurvivorCounter survivors;
};

struct DeviceSurvivors {
    explicit DeviceSurvivors(size_t size)
        : numerators(size), denominators(size) {}

    DeviceBuffer<long long> numerators;
    DeviceBuffer<int> denominators;
};

__device__ __forceinline__ unsigned long long reserve_survivor(
        unsigned long long *count, unsigned *overflow) {
    unsigned long long current = atomicCAS(count, 0ULL, 0ULL);
    while (current != ULLONG_MAX) {
        unsigned long long previous = atomicCAS(count, current, current + 1);
        if (previous == current) {
            return current;
        }
        current = previous;
    }
    atomicExch(overflow, 1U);
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
        uint32_t surviving, long long word, long long bound, int denominator,
        long long *__restrict__ output_numerators,
        int *__restrict__ output_denominators,
        unsigned long long *__restrict__ count,
        unsigned *__restrict__ overflow, unsigned long long capacity) {
    while (surviving) {
        int bit = __ffs(surviving) - 1;
        surviving &= surviving - 1;
        long long numerator = -bound + 32LL * word + bit;
        if (numerator <= bound) {
            unsigned long long index = reserve_survivor(count, overflow);
            if (index < capacity) {
                output_numerators[index] = numerator;
                output_denominators[index] = denominator;
            }
        }
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

template <typename Word>
__global__ void sieve_kernel(const uint32_t *__restrict__ masks,
                             const int *__restrict__ primes,
                             const long long *__restrict__ offsets,
                             const int *__restrict__ selected,
                             Word words, long long bound,
                             int min_denominator,
                             long long *__restrict__ output_numerators,
                             int *__restrict__ output_denominators,
                             unsigned long long *__restrict__ count,
                             unsigned *__restrict__ overflow,
                             unsigned long long capacity) {
#ifdef SHARED_MASK_ROWS
    extern __shared__ uint32_t local_masks[];
    __shared__ int local_row_offsets[kInitialPrimeCount];
#endif
    __shared__ int local_primes[kPrimeCount];
    __shared__ int denominator_residues[kPrimeCount];
    __shared__ long long local_offsets[kPrimeCount];
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
#pragma unroll
    for (int i = 0; i < kInitialPrimeCount; ++i) {
#ifdef SHARED_MASK_ROWS
        residues[i] = local_row_offsets[i] + threadIdx.x;
#else
        residues[i] = threadIdx.x;
#endif
    }
    for (Word word = threadIdx.x; word < words; word += kBlockSize) {
        uint32_t surviving = 0xffffffffu;
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
        emit_survivors(surviving, word, bound, denominator,
            output_numerators, output_denominators, count, overflow, capacity);
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
    check_cuda(cudaFuncSetAttribute(sieve_kernel<unsigned>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   plan.shared_mask_bytes), "cudaFuncSetAttribute");
    check_cuda(cudaFuncSetAttribute(sieve_kernel<long long>,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   plan.shared_mask_bytes), "cudaFuncSetAttribute");
#else
    (void)plan;
#endif
}

template <typename Word>
void launch_sieve_kernel(const SievePlan &plan, DeviceWorkspace &workspace,
                         DeviceSurvivors *survivors,
                         unsigned long long capacity, Word words) {
    long long *numerators = survivors ? survivors->numerators.data() : nullptr;
    int *denominators = survivors ? survivors->denominators.data() : nullptr;
#ifdef SHARED_MASK_ROWS
    sieve_kernel<Word><<<plan.denominator_range.count(), kBlockSize,
        plan.shared_mask_bytes>>>(
#else
    sieve_kernel<Word><<<plan.denominator_range.count(), kBlockSize>>>(
#endif
        workspace.masks.data(), workspace.primes.data(),
        workspace.offsets.data(), workspace.selected.data(), words,
        plan.numerator_bound, plan.denominator_range.first, numerators,
        denominators,
        workspace.survivors.count.data(), workspace.survivors.overflow.data(),
        capacity);
}

void launch_sieve(const SievePlan &plan, DeviceWorkspace &workspace,
                  DeviceSurvivors *survivors,
                  unsigned long long capacity) {
    workspace.survivors.clear();
    if (plan.word_count
        <= static_cast<long long>(UINT32_MAX) - (kBlockSize - 1)) {
        launch_sieve_kernel(plan, workspace, survivors, capacity,
                            static_cast<unsigned>(plan.word_count));
    } else {
        launch_sieve_kernel(plan, workspace, survivors, capacity,
                            plan.word_count);
    }
    check_cuda(cudaGetLastError(), "sieve_kernel launch");
}

std::vector<ModularCandidate> download_survivors(
        const DeviceSurvivors &survivors, size_t count) {
    std::vector<long long> numerators(count);
    std::vector<int> denominators(count);
    survivors.numerators.download(numerators);
    survivors.denominators.download(denominators);
    std::vector<ModularCandidate> candidates;
    candidates.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        candidates.push_back({numerators[i], denominators[i]});
    }
    return candidates;
}

}  // namespace

SieveResult run_modular_sieve(const Coefficients &coefficients,
                              long long numerator_bound,
                              DenominatorRange denominators) {
    SievePlan plan(numerator_bound, denominators);
    DeviceWorkspace workspace(plan);
    configure_sieve_kernel(plan);

    EventInterval basis_timer;
    EventInterval sieve_timer;
    basis_timer.start();
    build_mask_basis(plan, workspace, coefficients);
    basis_timer.stop();

    sieve_timer.start();
    launch_sieve(plan, workspace, nullptr, 0);
    unsigned long long count = workspace.survivors.download_count();
    if (count > std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("survivor count exceeds host address space");
    }

    std::vector<ModularCandidate> candidates;
    if (count != 0) {
        DeviceSurvivors survivors(static_cast<size_t>(count));
        launch_sieve(plan, workspace, &survivors, count);
        unsigned long long emitted = workspace.survivors.download_count();
        sieve_timer.stop();
        if (emitted != count) {
            throw std::runtime_error(
                "survivor count changed between sieve passes");
        }
        candidates = download_survivors(
            survivors, static_cast<size_t>(count));
    } else {
        sieve_timer.stop();
    }

    SieveMetrics metrics;
    metrics.basis_ms = basis_timer.milliseconds();
    metrics.sieve_ms = sieve_timer.milliseconds();
    metrics.word_count = static_cast<unsigned long long>(plan.word_count);
    metrics.initial_mask_bytes =
        static_cast<long double>(denominators.count()) * plan.word_count
        * kInitialPrimeCount * sizeof(uint32_t);
    return {std::move(candidates), metrics};
}

}  // namespace ratpoints_gpu
