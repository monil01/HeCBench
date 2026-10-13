// https://github.com/boris-dimitrov/z4_planar_langford_multigpu
//
// Copyright 2017 Boris Dimitrov, Portola Valley, CA 94028.
// Questions? Contact http://www.facebook.com/boris
//
// This program counts all permutations of the sequence 1, 1, 2, 2, 3, 3, ..., n, n
// in which the two occurrences of each m are separated by precisely m other numbers,
// and lines connecting all (m, m) pairs can be drawn on the page without crossing.
//
// See http://www.dialectrix.com/langford.html ("Planar Solutions") or Knuth volume 4a
// page 3.  Todo: Provide better Knuth reference.
//
//
// THE ALGORITHM
//
// This program runs a depth-first-search aka backtracking algorithm which chooses
// to "open" or "close" a pair at each position, starting with position 0, and
// whether that pair would be connected from "below" or "above".  There are 4 choices
// for each of the 2*n positions, making it O(4^(2n)) with maximum stack depth 6*n.
//
// When choosing to "close" at position k, it locates the matching "open" at k',
// and computes the distance m = k - k' + 1.  If this m has already been placed,
// closing at position k is not possible.
//
// When the number of open pairs reaches n, opening new pairs is no longer possible.
// Observing this constraint greatly prunes the search tree.
//
// The matching "open" at k' is very easy to find using two auxiliary stacks of
// currently open pairs, one for "below" and one for "above".
//
//
// DEDUPLICATION
//
// To dedup the Left <-> Right reversal symmetry, (1, 1) is placed in pos <= n.
//
// Many, but not all, top <-> bottom twins are deduped by forcing the pair in
// position 0 to be connected from below.
//
// Remaining duplicates are eliminated by storing all solutions in memory,
// with a final sort and count.  Fortunately, the number of solutions to the
// planar Langford problem is quite small, so this is feasible.
//
//
// IMPLEMENTATION TRICKS
//
// A nice boost in performance is realized through the use of a single 64-bit
// integer to encode the positions of *all* currently open pairs;  in this compact
// representation, we can quickly "pop" the position of the most recently open pair
// by using the operations
//
//     ffsl(x)       if x is non-zero, return one plus the index of the least
//                   significant bit of x;  if x is zero, return zero;
//                   result range 0..64
//
//     x &= (x-1)    clear the least signifficant 1-bit of x
//
//
// EXAMPLE OUTPUT
//
//     1488034642458 Computing PL(2, 19)
//     1488034642458 Will use 2 system GPU(s).
//     1488034642463 GPU 1 init took 0.005 seconds.
//     1488034642463 GPU 0 init took 0.005 seconds.
//     1488034652163 GPU 0 computation took 9.70008 seconds on GPU clock, 9.7 seconds on host clock.
//     1488034652178 GPU 1 computation took 9.71449 seconds on GPU clock, 9.715 seconds on host clock.
//     1488034652229 Result 2384 for n = 19 MATCHES previously published result
//
// The first number on each output line is a unix timestamp, i.e., milliseconds elapsed
// since Jan 1, 1970 GMT.  You may convert it to human-readable datetime using python,
// as follows.
//
// 1) Start "python"
// 2) Type "import time" and press enter
// 3) Type "time.localtime(1488034652163 / 1000.0)" and press enter
//
// The result is a decoding of unix timestamp 1488034652163 in your local time zone:
//
//     time.struct_time(tm_year=2017, tm_mon=2, tm_mday=25, tm_hour=6,
//         tm_min=57, tm_sec=32, tm_wday=5, tm_yday=56, tm_isdst=0)
//
//
// PRINTING ALL SOLUTION SEQUENCES
//
// If you wish all solutions sequences printed, change 'kPrint' below to 'true',
// and recompile.
//
//
// ACHIEVEMENTS
//
// On March 6, 2017 at 11:38am PST this program computed PL(2, 28) after very close to 168 hours
// of work on a pair of NVIDIA Titan X Pascal GPUs in a single workstation.
//
//     1488221331811 Computing PL(2, 28)
//     1488221331811 Will use 2 system GPU(s).
//     1488221331817 GPU 1 init took 0.006 seconds.
//     1488221331817 GPU 0 init took 0.006 seconds.
//     1488823122601 GPU 1 computation took 601769 seconds on GPU clock, 601791 seconds on host clock.
//     1488829105991 GPU 0 computation took 607752 seconds on GPU clock, 607774 seconds on host clock.
//     1488829106231 Result 817717 for n = 28 MATCHES previously published result
//
// The GPU clock ran at ~1835 MHz, with each GPU consuming ~150 watts, at temperature ~62 Celsius,
// during most of that 7 day computation.  The last few hours the chips were less busy while
// finishing up a few of the longest running threads, and consumed correspondingly less power.
//
// SEE ALSO
//
// There is a variant of this program that runs on CPUs.  Performance is comparable between a single
// Titan X Pascal GPU (16nm process, mid 2016) and a 22-core Xeon E5-2699v4 (14nm, early 2016).
//
//     https://github.com/boris-dimitrov/z4_planar_langford

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>
#include "dmem_runtime.hpp"
using namespace std;

// to avoid integer overflow, n should not exceed this constant
constexpr int kMaxN = 31;

// kLimit = 100 million means we can use up to 3.2GB of RAM on each GPU to store
// results before sorting;  if more memory is needed, we detect and bail
constexpr int64_t kLimit = 100000000;

// set to "true" if you want each solution printed
constexpr bool kPrint = false;

static_assert(sizeof(int64_t) == 8, "int64_t is not 8 bytes");
static_assert(sizeof(int32_t) == 4, "int32_t is not 4 bytes");
static_assert(sizeof(int8_t) == 1, "int64_t is not 1 byte");

constexpr int64_t lsb = 1;
constexpr int32_t lsb32 = 1;

constexpr int div_up(int p, int q) {
  return (p + (q - 1)) / q;
};

// This type represents a solution by letting pos[m-1] be
// the position of the closing m, for m = 1, 2, ... n.
template <int n>
using Positions = std::array<int8_t, n>;

// All planar sequences (including duplicates) are stored in a vector for sorting.
template <int n>
using Results = std::vector<Positions<n>>;

// 8-byte alignment probably helps with concurrent writes to adjacent instances
template <int n>
using PositionsGPUAligned = int64_t[(n + 7) / 8];

template <int n>
using PositionsGPU = int8_t[div_up(n, 8) * 8];

// at depth k in the search, the bits of availability[k+1] represent the still
// unused m;  thus helping ensure that each (m, m) pair is placed just once
template <int n>
using Availability = int32_t[2 * n + 1];

// at depth k in the search, the bits of open[2*k+2] represent the positions
// of all pairs that are open above;  open[2*k+3] is the same for below;
// k ranges over 0, 1, ..., 2n-1
template <int n>
using Open = int64_t[4 * n + 2];

template <int n>
using Stack = int8_t[24 * n];

template <int n>
void print(const Positions<n>& pos);

// For some reason, 4 is the magic number that gives us best perf in this algorithm
constexpr int kThreadsPerBlock = 4;

// subdivide the search tree for this many logical threads
// due to the naive math below, numbers like 2^r-1 work much better than 2^r
constexpr int kNumLogicalThreads = 16383;

// Sort the vector of solution sequences and count the unique ones.
// Optionally print each unique one.
template <int n>
int64_t unique_count(Results<n> &results) {
  int64_t total = results.size();
  int64_t unique = total;
  sort(results.begin(), results.end());
  if (kPrint && total) {
    print<n>(results[0]);
  }
  for (int i=1; i<total; ++i) {
    if (results[i] == results[i-1]) {
      --unique;
    } else if (kPrint) {
      print<n>(results[i]);
    }
  }
  return unique;
}

// Return number of milliseconds elapsed since Jan 1, 1970 00:00 GMT.
long unixtime() {
  using namespace chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

// Start and manage the computation on GPU device "device"
template <int n>
void run_gpu_d(Results<n>& final_results) {
  assert(sizeof(int64_t) == 8);

  int64_t count = 0;
  constexpr int64_t kAlignedCnt = (n + 7) / 8;

  int64_t* results_device;
  iris_alloc((void**)&results_device, (1 + kLimit * kAlignedCnt) * sizeof(int64_t));
  iris_copy(results_device, &count, sizeof(int64_t), iris_h2d);
  int blocks_x = div_up(kNumLogicalThreads, kThreadsPerBlock);
  dim3 blocks(blocks_x);

  iris_sync();
  auto start = std::chrono::steady_clock::now();

  const char* name = n==7 ? "dfs_gpu_7" : n==8 ? "dfs_gpu_8" : n==11 ? "dfs_gpu_11" : n==12 ? "dfs_gpu_12" : "dfs_gpu_15";
  iris_launch(name, dim3(blocks), dim3(kThreadsPerBlock), results_device);

  iris_sync();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  cout << "Kernel execution time:  " << time * 1e-9f << " (s)\n";

  iris_copy(&count, results_device, sizeof(int64_t), iris_d2h);

  if (count >= kLimit) {
    cout << "Result for n = " << n << " will be bogus because GPU "
      << " exceeded " << kLimit << " solutions.\n";
  }
  int64_t data_size = (1 + count * kAlignedCnt) * sizeof(int64_t);

  int64_t* results_host = (int64_t*) malloc (data_size);
  iris_copy(results_host, results_device, data_size, iris_d2h);
  iris_free(results_device);

  // Compact results into a vector
  for (int i=0; i<count; ++i) {
    Positions<n> pos;
    PositionsGPU<n>& gpos = *((PositionsGPU<n>*)(results_host + 1 + (kAlignedCnt * i)));
    for (int j=0; j<n; ++j) {
      pos[j] = gpos[j];
    }
    final_results.push_back(pos);
  }
  free(results_host);
}

// Start a CPU thread to manage each GPU device and wait for the computation to end.
template <int n>
void run_gpu(const int64_t* known_results) {
  cout << "\n";
  cout << "------\n";
  cout << unixtime() << " Computing PL(2, " << n << ")\n";
  if (n > kMaxN) {
    cout << unixtime() << " Sorry, n = " << n << " exceeds the max allowed " << kMaxN << "\n";
    return;
  }

  int64_t total;
  Results<n> final_results;

  run_gpu_d<n>(final_results);

  // Sort and unique count on CPU.
  total = unique_count<n>(final_results);
  cout << unixtime() << " Result " << total << " for n = " << n;
  if (n < 0 || n >= 64 || known_results[n] == -1) {
    cout << " is NEW";
  } else if (known_results[n] == total) {
    cout << " MATCHES previously published result";
  } else {
    cout << " MISMATCHES previously published result " << known_results[n];
  }
  cout << "\n------\n\n";
}

void init_known_results(int64_t (&known_results)[64]) {
  for (int i=0;  i<64; ++i) {
    known_results[i] = 0;
  }
  // There are no published results for n >= 29
  for (int i = 29;  i<64;  ++i) {
    if (i % 4 == 3 || i % 4 == 0) {
      known_results[i] = -1;
    }
  }
  known_results[3]  = 1;
  known_results[4]  = 0;
  known_results[7]  = 0;
  known_results[8]  = 4;
  known_results[11] = 16;
  known_results[12] = 40;
  known_results[15] = 194;
  known_results[16] = 274;
  known_results[19] = 2384;
  known_results[20] = 4719;
  known_results[23] = 31856;
  known_results[24] = 62124;
  known_results[27] = 426502;
  known_results[28] = 817717;
}

template <int n>
void print(const Positions<n>& pos) {
  cout << unixtime() << " Sequence ";
  int s[2 * n];
  for (int i=0; i<2*n; ++i) {
    s[i] = -1;
  }
  for (int m=1;  m<=n;  ++m) {
    int k2 = pos[m-1];
    int k1 = k2 - m - 1;
    assert(0 <= k1);
    assert(k2 < 2*n);
    assert(s[k1] == -1);
    assert(s[k2] == -1);
    s[k1] = s[k2] = m;
  }
  for (int i=0;  i<2*n;  ++i) {
    const int64_t m = s[i];
    assert(0 <= m);
    assert(m <= n);
    cout << setw(3) << m;
  }
  cout << "\n";
}

int main(int argc, char **argv) {
  int64_t known_results[64];
  init_known_results(known_results);
  /* we cannot do 3 and 4 anymore due to unrolling
     run_gpu<3>(known_results);
     run_gpu<4>(known_results);
   */
  run_gpu<7>(known_results);
  run_gpu<8>(known_results);
  run_gpu<11>(known_results);
  run_gpu<12>(known_results);
  run_gpu<15>(known_results);
  //run_gpu<16>(known_results);
  //run_gpu<19>(known_results);
  //run_gpu<20>(known_results);
  //run_gpu<23>(known_results);
  //run_gpu<24>(known_results);
  //run_gpu<27>(known_results);
  //run_gpu<28>(known_results);
  iris_finish();
  return 0;
}
