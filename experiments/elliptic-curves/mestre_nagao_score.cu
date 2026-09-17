#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(2); \
    } \
} while (0)

struct Candidate {
    long long id;
    long long num;
    long long den;
};

__device__ __forceinline__ int norm64(long long value, int p) {
    long long r = value % p;
    return int(r < 0 ? r + p : r);
}

__device__ __forceinline__ int addm(int a, int b, int p) {
    int v = a + b;
    if (v >= p) v -= p;
    return v;
}

__device__ __forceinline__ int subm(int a, int b, int p) {
    int v = a - b;
    if (v < 0) v += p;
    return v;
}

__device__ __forceinline__ int mulm(int a, int b, int p) {
    return int((static_cast<long long>(a) * static_cast<long long>(b)) % p);
}

__device__ int powm(int base, int exp, int p) {
    int result = 1;
    while (exp) {
        if (exp & 1) result = mulm(result, base, p);
        base = mulm(base, base, p);
        exp >>= 1;
    }
    return result;
}

__device__ __forceinline__ int madd(int acc, long long coeff, int value, int p) {
    int c = norm64(coeff, p);
    return addm(acc, mulm(c, value, p), p);
}

__global__ void score_kernel(
    const Candidate* candidates,
    int candidate_count,
    const int* primes,
    const size_t* offsets,
    int prime_count,
    const signed char* chi,
    double* scores,
    int* good_counts,
    int* counts)
{
    int ci = blockIdx.x;
    if (ci >= candidate_count) return;
    int tid = threadIdx.x;
    extern __shared__ int shared[];

    Candidate c = candidates[ci];
    double score = scores[ci];
    int good = good_counts[ci];

    for (int pi = 0; pi < prime_count; ++pi) {
        int prime = primes[pi];
        int den = norm64(c.den, prime);
        if (den == 0) {
            if (tid == 0 && counts) counts[size_t(ci) * prime_count + pi] = -1;
            __syncthreads();
            continue;
        }
        int t = mulm(norm64(c.num, prime), powm(den, prime - 2, prime), prime);
        int t2 = mulm(t, t, prime);
        int t3 = mulm(t2, t, prime);
        int t4 = mulm(t2, t2, prime);

        int L = 0;
        L = madd(L, 446667LL, t2, prime);
        L = madd(L, 471466LL, t, prime);
        L = addm(L, norm64(239031LL, prime), prime);

        int pv = 0;
        pv = madd(pv, 318552LL, t2, prime);
        pv = madd(pv, 368554LL, t, prime);
        pv = addm(pv, norm64(-72570LL, prime), prime);

        int qv = 0;
        qv = madd(qv, 733413LL, t2, prime);
        qv = madd(qv, -45082LL, t, prime);
        qv = addm(qv, norm64(-14960LL, prime), prime);

        int Dinside = 0;
        Dinside = madd(Dinside, 7174492962LL, t4, prime);
        Dinside = madd(Dinside, -7114589515LL, t3, prime);
        Dinside = madd(Dinside, -22069002960LL, t2, prime);
        Dinside = madd(Dinside, 3909144679LL, t, prime);
        Dinside = addm(Dinside, norm64(-205134150LL, prime), prime);
        int D = mulm(norm64(5LL, prime), Dinside, prime);

        int E = 0;
        E = madd(E, 882769396002LL, t4, prime);
        E = madd(E, 811447034567LL, t3, prime);
        E = madd(E, -1174040743LL, t2, prime);
        E = madd(E, -32493137198LL, t, prime);
        E = addm(E, norm64(-2386325360LL, prime), prime);

        int B = mulm(mulm(pv, qv, prime), addm(addm(L, pv, prime), qv, prime), prime);
        B = subm(B, mulm(pv, E, prime), prime);
        B = subm(B, mulm(qv, D, prime), prime);

        int a1 = L ? prime - L : 0;
        int a2 = addm(D, E, prime);
        int a3 = B ? prime - B : 0;
        int a4 = mulm(D, E, prime);
        int b2 = addm(mulm(a1, a1, prime), mulm(norm64(4LL, prime), a2, prime), prime);
        int b4 = addm(mulm(norm64(2LL, prime), a4, prime), mulm(a1, a3, prime), prime);
        int b6 = mulm(a3, a3, prime);
        int b8 = subm(mulm(a2, mulm(a3, a3, prime), prime),
                       mulm(a1, mulm(a3, a4, prime), prime), prime);
        b8 = subm(b8, mulm(a4, a4, prime), prime);

        int delta = 0;
        delta = subm(delta, mulm(mulm(b2, b2, prime), b8, prime), prime);
        delta = subm(delta, mulm(norm64(8LL, prime), mulm(mulm(b4, b4, prime), b4, prime), prime), prime);
        delta = subm(delta, mulm(norm64(27LL, prime), mulm(b6, b6, prime), prime), prime);
        delta = addm(delta, mulm(norm64(9LL, prime), mulm(mulm(b2, b4, prime), b6, prime), prime), prime);
        if (delta == 0) {
            if (tid == 0 && counts) counts[size_t(ci) * prime_count + pi] = -1;
            __syncthreads();
            continue;
        }

        int local = 0;
        for (int x = tid; x < prime; x += blockDim.x) {
            int A = addm(mulm(L, x, prime), B, prime);
            int rhs = mulm(x, mulm(addm(x, D, prime), addm(x, E, prime), prime), prime);
            int disc = addm(mulm(A, A, prime), mulm(norm64(4LL, prime), rhs, prime), prime);
            local += int(chi[offsets[pi] + disc]);
        }
        shared[tid] = local;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (tid < stride) shared[tid] += shared[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            int np = prime + 1 + shared[0];
            if (counts) counts[size_t(ci) * prime_count + pi] = np;
            score += (1.0 - double(prime - 1) / double(np)) * log(double(prime));
            ++good;
        }
        __syncthreads();
    }
    if (tid == 0) {
        scores[ci] = score;
        good_counts[ci] = good;
    }
}

static std::vector<int> primes_up_to(int bound) {
    std::vector<bool> sieve(bound + 1, true);
    if (bound >= 0) sieve[0] = false;
    if (bound >= 1) sieve[1] = false;
    for (int i = 2; i * i <= bound; ++i) {
        if (sieve[i]) for (int j = i * i; j <= bound; j += i) sieve[j] = false;
    }
    std::vector<int> out;
    for (int i = 3; i <= bound; i += 2) if (sieve[i]) out.push_back(i);
    return out;
}

static size_t checked_add(size_t a, size_t b, const char* label) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::runtime_error(std::string(label) + " overflows size_t");
    return a + b;
}

static size_t checked_multiply(size_t a, size_t b, const char* label) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::runtime_error(std::string(label) + " overflows size_t");
    return a * b;
}

struct PrimeChunk { size_t begin, end, chi_bytes; };

static std::vector<PrimeChunk> plan_chunks(const std::vector<int>& primes,
                                            size_t budget, size_t& total_chi_bytes) {
    std::vector<PrimeChunk> chunks;
    total_chi_bytes = 0;
    size_t begin = 0, bytes = 0;
    for (size_t i = 0; i < primes.size(); ++i) {
        size_t p = static_cast<size_t>(primes[i]);
        total_chi_bytes = checked_add(total_chi_bytes, p, "total chi entries");
        if (p > budget)
            throw std::runtime_error("chi chunk budget is smaller than prime " + std::to_string(p));
        if (bytes && p > budget - bytes) {
            chunks.push_back({begin, i, bytes});
            begin = i;
            bytes = 0;
        }
        bytes = checked_add(bytes, p, "chunk chi entries");
    }
    if (bytes) chunks.push_back({begin, primes.size(), bytes});
    return chunks;
}

int main(int argc, char** argv) {
    try {
    int device = -1;
    int prime_bound = -1;
    size_t chi_chunk_bytes = 256ULL * 1024 * 1024;
    bool plan_only = false;
    std::string input_path, output_path, counts_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) device = std::stoi(argv[++i]);
        else if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--prime-bound" && i + 1 < argc) prime_bound = std::stoi(argv[++i]);
        else if (arg == "--counts" && i + 1 < argc) counts_path = argv[++i];
        else if (arg == "--chi-chunk-bytes" && i + 1 < argc) chi_chunk_bytes = std::stoull(argv[++i]);
        else if (arg == "--plan-only") plan_only = true;
        else { std::cerr << "unknown/incomplete argument: " << arg << "\n"; return 2; }
    }
    if (prime_bound < 3 || prime_bound > 1000000)
        throw std::runtime_error("--prime-bound must be between 3 and 1000000");
    if (chi_chunk_bytes == 0 || chi_chunk_bytes > (1ULL << 30))
        throw std::runtime_error("--chi-chunk-bytes must be between 1 byte and 1 GiB");
    std::vector<int> primes = primes_up_to(prime_bound);
    size_t total_chi_bytes = 0;
    std::vector<PrimeChunk> chunks = plan_chunks(primes, chi_chunk_bytes, total_chi_bytes);
    size_t max_chi_bytes = 0, max_chunk_primes = 0;
    for (const PrimeChunk& chunk : chunks) {
        max_chi_bytes = std::max(max_chi_bytes, chunk.chi_bytes);
        max_chunk_primes = std::max(max_chunk_primes, chunk.end - chunk.begin);
    }
    if (plan_only) {
        std::cout << "prime_bound=" << prime_bound << " primes=" << primes.size()
                  << " chunks=" << chunks.size() << " max_chi_bytes=" << max_chi_bytes
                  << " total_chi_entries=" << total_chi_bytes << "\n";
        return 0;
    }
    if (device < 0 || input_path.empty() || output_path.empty())
        throw std::runtime_error("usage: helper --device N --input FILE --output FILE --prime-bound B [--counts FILE] [--chi-chunk-bytes N] [--plan-only]");

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::ifstream input(input_path);
    if (!input) throw std::runtime_error("cannot open input");
    std::vector<Candidate> candidates;
    Candidate c{};
    while (input >> c.id >> c.num >> c.den) {
        if (c.den <= 0) throw std::runtime_error("candidate denominator must be positive");
        candidates.push_back(c);
    }
    if (candidates.empty()) throw std::runtime_error("no candidates");
    if (candidates.size() > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        candidates.size() > static_cast<size_t>(prop.maxGridSize[0]))
        throw std::runtime_error("candidate count exceeds one-dimensional CUDA grid limit");

    Candidate* d_candidates = nullptr;
    int* d_primes = nullptr;
    size_t* d_offsets = nullptr;
    signed char* d_chi = nullptr;
    double* d_scores = nullptr;
    int* d_good = nullptr;
    int* d_counts = nullptr;
    size_t nc = candidates.size();
    size_t np = primes.size();
    size_t candidate_bytes = checked_multiply(nc, sizeof(Candidate), "candidate allocation");
    size_t score_bytes = checked_multiply(nc, sizeof(double), "score allocation");
    size_t good_bytes = checked_multiply(nc, sizeof(int), "good-count allocation");
    size_t chunk_prime_bytes = checked_multiply(max_chunk_primes, sizeof(int), "prime allocation");
    size_t chunk_offset_bytes = checked_multiply(max_chunk_primes + 1, sizeof(size_t), "offset allocation");
    size_t chunk_count_bytes = checked_multiply(checked_multiply(nc, max_chunk_primes, "chunk count entries"),
                                                sizeof(int), "chunk count allocation");
    size_t all_count_bytes = 0;
    if (!counts_path.empty()) {
        all_count_bytes = checked_multiply(checked_multiply(nc, np, "all count entries"),
                                           sizeof(int), "all count allocation");
        if (all_count_bytes > (1ULL << 30))
            throw std::runtime_error("--counts would require over 1 GiB of host storage; use fewer candidates");
    }
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t needed = checked_add(max_chi_bytes, candidate_bytes, "GPU allocation estimate");
    needed = checked_add(needed, score_bytes, "GPU allocation estimate");
    needed = checked_add(needed, good_bytes, "GPU allocation estimate");
    needed = checked_add(needed, chunk_prime_bytes, "GPU allocation estimate");
    needed = checked_add(needed, chunk_offset_bytes, "GPU allocation estimate");
    if (!counts_path.empty()) needed = checked_add(needed, chunk_count_bytes, "GPU allocation estimate");
    if (needed > free_bytes / 2)
        throw std::runtime_error("chunk and candidate buffers exceed half of free GPU memory; reduce --chi-chunk-bytes or candidate batch size");

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_candidates), candidate_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_primes), chunk_prime_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_offsets), chunk_offset_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_chi), max_chi_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scores), score_bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_good), good_bytes));
    if (!counts_path.empty()) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_counts), chunk_count_bytes));
    }

    CUDA_CHECK(cudaMemcpy(d_candidates, candidates.data(), candidate_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_scores, 0, score_bytes));
    CUDA_CHECK(cudaMemset(d_good, 0, good_bytes));
    std::vector<int> all_counts;
    if (!counts_path.empty()) all_counts.resize(all_count_bytes / sizeof(int));
    double kernel_seconds = 0.0;
    for (const PrimeChunk& chunk : chunks) {
        size_t chunk_np = chunk.end - chunk.begin;
        std::vector<size_t> offsets(chunk_np + 1, 0);
        for (size_t j = 0; j < chunk_np; ++j)
            offsets[j + 1] = checked_add(offsets[j], static_cast<size_t>(primes[chunk.begin + j]), "chunk offset");
        if (offsets.back() != chunk.chi_bytes) throw std::runtime_error("chunk plan offset mismatch");
        std::vector<signed char> chi(chunk.chi_bytes, -1);
        for (size_t j = 0; j < chunk_np; ++j) {
            int p = primes[chunk.begin + j];
            size_t off = offsets[j];
            chi[off] = 0;
            for (int y = 1; y < p; ++y)
                chi[off + (static_cast<long long>(y) * y) % p] = 1;
        }
        CUDA_CHECK(cudaMemcpy(d_primes, primes.data() + chunk.begin,
                              chunk_np * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_chi, chi.data(), chi.size(), cudaMemcpyHostToDevice));
        auto kernel_started = std::chrono::steady_clock::now();
        score_kernel<<<int(nc), 256, 256 * sizeof(int)>>>(
            d_candidates, int(nc), d_primes, d_offsets, int(chunk_np), d_chi,
            d_scores, d_good, d_counts);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        kernel_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - kernel_started).count();
        if (d_counts) {
            std::vector<int> chunk_counts(nc * chunk_np);
            CUDA_CHECK(cudaMemcpy(chunk_counts.data(), d_counts, chunk_counts.size() * sizeof(int), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < nc; ++i)
                std::copy(chunk_counts.begin() + i * chunk_np,
                          chunk_counts.begin() + (i + 1) * chunk_np,
                          all_counts.begin() + i * np + chunk.begin);
        }
    }

    std::vector<double> scores(nc);
    std::vector<int> good(nc);
    CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, score_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(good.data(), d_good, good_bytes, cudaMemcpyDeviceToHost));

    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("cannot open score output");
    output << std::setprecision(17);
    for (size_t i = 0; i < nc; ++i) {
        output << candidates[i].id << '\t' << scores[i] << '\t' << good[i] << '\n';
    }

    if (!counts_path.empty()) {
        std::ofstream count_out(counts_path);
        if (!count_out) throw std::runtime_error("cannot open count output");
        for (size_t i = 0; i < nc; ++i) {
            for (size_t j = 0; j < np; ++j) {
                count_out << candidates[i].id << '\t' << primes[j] << '\t'
                          << all_counts[i * np + j] << '\n';
            }
        }
    }

    std::cerr << "device=" << device << " name=\"" << prop.name << "\" candidates=" << nc
              << " primes=" << np << " prime_bound=" << prime_bound
              << " prime_chunks=" << chunks.size() << " max_chi_bytes=" << max_chi_bytes
              << " total_chi_entries=" << total_chi_bytes
              << " kernel_seconds=" << std::fixed << std::setprecision(6) << kernel_seconds << "\n";

    if (d_counts) cudaFree(d_counts);
    cudaFree(d_good); cudaFree(d_scores); cudaFree(d_chi); cudaFree(d_offsets);
    cudaFree(d_primes); cudaFree(d_candidates);
    return 0;
    } catch (const std::exception& exc) {
        std::cerr << "mestre_nagao_score failed: " << exc.what() << "\n";
        return 2;
    }
}
