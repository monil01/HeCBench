/*------------------------------------------------------------------------------
* Copyright 2015-16: Tom Deakin, Simon McIntosh-Smith, University of Bristol HPC
* Based on John D. McCalpin’s original STREAM benchmark for CPUs
*------------------------------------------------------------------------------
* License:
*  1. You are free to use this program and/or to redistribute
*     this program.
*  2. You are free to modify this program for your own use,
*     including commercial use, subject to the publication
*     restrictions in item 3.
*  3. You are free to publish results obtained from running this
*     program, or from works that you derive from this program,
*     with the following limitations:
*     3a. In order to be referred to as "BabelStream benchmark results",
*         published results must be in conformance to the BabelStream
*         Run Rules published at
*         http://github.com/UoB-HPC/BabelStream/wiki/Run-Rules
*         and incorporated herein by reference.
*         The copyright holders retain the
*         right to determine conformity with the Run Rules.
*     3b. Results based on modified source code or on runs not in
*         accordance with the BabelStream Run Rules must be clearly
*         labelled whenever they are published.  Examples of
*         proper labelling include:
*         "tuned BabelStream benchmark results"
*         "based on a variant of the BabelStream benchmark code"
*         Other comparable, clear and reasonable labelling is
*         acceptable.
*     3c. Submission of results to the BabelStream benchmark web site
*         is encouraged, but not required.
*  4. Use of this program or creation of derived works based on this
*     program constitutes acceptance of these licensing restrictions.
*  5. Absolutely no warranty is expressed or implied.
*/

#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <iris/iris.hpp>
#include <sstream>
#include <stdexcept>
#include <cstdlib>


// Thread block size
#define TBSIZE 256

// Number of blocks for the DOT kernel 
#define DOT_NUM_BLOCKS 256

// scalar constanst for the mul, triad and nstream kernels
#define SCALAR (0.4)

int ARRAY_SIZE = 33554432; // Default size of 2^25
unsigned int num_times = 100;


void require_iris(int status) {
  if (status != IRIS_SUCCESS) throw std::runtime_error("IRIS operation failed");
}
int selected_device = -1;

template<class T>
const char* kernel_name(const char* operation) {
  static thread_local std::string name;
  name = std::string(operation) + (sizeof(T) == sizeof(float) ? "_float" : "_double");
  return name.c_str();
}
template<class T>
void init_arrays(iris::DMem& a, iris::DMem& b, iris::DMem& c, T x, T y, T z) {
  iris::Task task;
  void* p[] = { &a, &b, &c, &x, &y, &z };
  int info[] = { iris_w, iris_w, iris_w, sizeof(T), sizeof(T), sizeof(T) };
  size_t gws = ARRAY_SIZE, lws = TBSIZE;
  require_iris(task.kernel(kernel_name<T>("init"), 1, nullptr, &gws, &lws, 6, p, info));
  require_iris(task.submit(selected_device, nullptr, true));
}
template<class T>
void operation(const char* name, iris::DMem& a, iris::DMem& b, iris::DMem* c,
               int mode_a, int mode_b, int mode_c) {
  iris::Task task;
  void* p[] = { &a, &b, c };
  int info[] = { mode_a, mode_b, mode_c };
  size_t gws = ARRAY_SIZE, lws = TBSIZE;
  require_iris(task.kernel(kernel_name<T>(name), 1, nullptr, &gws, &lws, c ? 3 : 2, p, info));
  require_iris(task.submit(selected_device, nullptr, true));
}
template<class T> void copy(iris::DMem& a, iris::DMem& c) {
  operation<T>("copy", a, c, nullptr, iris_r, iris_w, 0);
}
template<class T> void mul(iris::DMem& b, iris::DMem& c) {
  operation<T>("mul", b, c, nullptr, iris_w, iris_r, 0);
}
template<class T> void add(iris::DMem& a, iris::DMem& b, iris::DMem& c) {
  operation<T>("add", a, b, &c, iris_r, iris_r, iris_w);
}
template<class T> void triad(iris::DMem& a, iris::DMem& b, iris::DMem& c) {
  operation<T>("triad", a, b, &c, iris_w, iris_r, iris_r);
}
template<class T> void nstream(iris::DMem& a, iris::DMem& b, iris::DMem& c) {
  operation<T>("nstream", a, b, &c, iris_rw, iris_r, iris_r);
}
template<class T> T dot(iris::DMem& a, iris::DMem& b, iris::DMem& sums_mem, T* sums) {
  iris::Task task;
  void* p[] = { &a, &b, &sums_mem, &ARRAY_SIZE };
  int info[] = { iris_r, iris_r, iris_w, sizeof(ARRAY_SIZE) };
  size_t gws = DOT_NUM_BLOCKS * TBSIZE, lws = TBSIZE;
  require_iris(task.kernel(kernel_name<T>("dot"), 1, nullptr, &gws, &lws, 4, p, info));
  require_iris(task.flush_out(sums_mem));
  require_iris(task.submit(selected_device, nullptr, true));
  T sum = 0;
  for (int i = 0; i < DOT_NUM_BLOCKS; ++i) sum += sums[i];
  return sum;
}
// Runs the kernel(s) and prints output.
template <typename T>
void run()
{
  std::streamsize ss = std::cout.precision();

  std::cout << "Running kernels " << num_times << " times" << std::endl;

  // The array size must be divisible by TBSIZE for kernel launches
  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

  std::vector<T> storage_a(ARRAY_SIZE), storage_b(ARRAY_SIZE), storage_c(ARRAY_SIZE);
  std::vector<T> partial_sums(DOT_NUM_BLOCKS);
  T* sums = partial_sums.data();
  iris::DMem da(storage_a.data(), ARRAY_SIZE * sizeof(T));
  iris::DMem db(storage_b.data(), ARRAY_SIZE * sizeof(T));
  iris::DMem dc(storage_c.data(), ARRAY_SIZE * sizeof(T));
  iris::DMem dsum(sums, DOT_NUM_BLOCKS * sizeof(T));

  if (sizeof(T) == sizeof(float))
    std::cout << "Precision: float" << std::endl;
  else
    std::cout << "Precision: double" << std::endl;

  // MB = 10^6
  std::cout << std::setprecision(1) << std::fixed
    << "Array size: " << ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
    << " (=" << ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
  std::cout << "Total size: " << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-6 << " MB"
    << " (=" << 3.0*ARRAY_SIZE*sizeof(T)*1.0E-9 << " GB)" << std::endl;
  std::cout.precision(ss);

  // Initialize device arrays
  init_arrays(da, db, dc, (T)0.1, (T)0.2, T(0.0));

  // simulation-based validation
  std::vector<T> ha (ARRAY_SIZE, (T)0.1);
  std::vector<T> hb (ARRAY_SIZE, (T)0.2);
  std::vector<T> hc (ARRAY_SIZE);
  std::vector<T> hd (ARRAY_SIZE); // nstream result

  copy<T>(da, dc); // c = a
  mul<T>(db, dc);  // b = c * scalar
  add<T>(da, db, dc); // c = a + b
  triad<T>(da, db, dc); // a = b + scalar * c
  T sum_d = dot<T>(da, db, dsum, sums); // s = dot(a * b)
  nstream<T>(da, db, dc); //  a += b + scalar * c

  for (int i = 0; i < ARRAY_SIZE; i++) {
    hc[i] = ha[i];
    hb[i] = hc[i] * SCALAR;
    hc[i] = ha[i] + hb[i];
    ha[i] = hb[i] + SCALAR * hc[i];
  }
  double sum_r = 0;
  for (int i = 0; i < ARRAY_SIZE; i++) sum_r += ha[i] * hb[i]; 
  for (int i = 0; i < ARRAY_SIZE; i++) ha[i] += hb[i] + SCALAR * hc[i];
  {
    iris::Task readback;
    require_iris(readback.flush_out(da));
    require_iris(readback.submit(selected_device, nullptr, true));
    hd = storage_a;
  }
  bool ok = true;
  if (std::fabs(sum_r - sum_d) >= 1) {
    std::cout << "dot: " << sum_r << " " << sum_d << std::endl;
    ok = false;
  }
  for (int i = 0; i < ARRAY_SIZE; i++) {
    if (std::fabs(hd[i] - ha[i]) > 1e-3) {
      std::cout << "a: " << hd[i] << " " << ha[i] << std::endl;
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS": "FAIL");

  // List of times
  std::vector<std::vector<double>> timings(6);

  // Declare timers
  std::chrono::high_resolution_clock::time_point t1, t2;

  // Last computed dot product, used for validation
  T sum = 0.0;

  // Main loop
  for (unsigned int k = 0; k < num_times; k++)
  {
    // Execute Copy
    t1 = std::chrono::high_resolution_clock::now();
    copy<T>(da, dc);
    t2 = std::chrono::high_resolution_clock::now();
    timings[0].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());


    // Execute Mul
    t1 = std::chrono::high_resolution_clock::now();
    mul<T>(db, dc);
    t2 = std::chrono::high_resolution_clock::now();
    timings[1].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Add
    t1 = std::chrono::high_resolution_clock::now();
    add<T>(da, db, dc);
    t2 = std::chrono::high_resolution_clock::now();
    timings[2].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Triad
    t1 = std::chrono::high_resolution_clock::now();
    triad<T>(da, db, dc);
    t2 = std::chrono::high_resolution_clock::now();
    timings[3].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute Dot
    t1 = std::chrono::high_resolution_clock::now();
    dot<T>(da, db, dsum, sums);
    t2 = std::chrono::high_resolution_clock::now();
    timings[4].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());

    // Execute NStream
    t1 = std::chrono::high_resolution_clock::now();
    nstream<T>(da, db, dc);
    t2 = std::chrono::high_resolution_clock::now();
    timings[5].push_back(std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count());
  }

  // Display timing results
  std::cout
    << std::left << std::setw(12) << "Function"
    << std::left << std::setw(12) << "MBytes/sec"
    << std::left << std::setw(12) << "Min (sec)"
    << std::left << std::setw(12) << "Max"
    << std::left << std::setw(12) << "Average"
    << std::endl
    << std::fixed;

  std::vector<std::string> labels;
  std::vector<size_t> sizes;

  labels = {"Copy", "Mul", "Add", "Triad", "Dot", "Nstream"};
  sizes = {
    2 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    3 * sizeof(T) * ARRAY_SIZE,
    2 * sizeof(T) * ARRAY_SIZE,
    4 * sizeof(T) * ARRAY_SIZE};

  for (size_t i = 0; i < timings.size(); ++i)
  {
    // Get min/max; ignore the first result
    auto minmax = std::minmax_element(timings[i].begin()+1, timings[i].end());

    // Calculate average; ignore the first result
    double average = std::accumulate(timings[i].begin()+1, timings[i].end(), 0.0) / (double)(num_times - 1);

    double bandwidth = 1.0E-6 * sizes[i] / (*minmax.first);

    std::cout
      << std::left << std::setw(12) << labels[i]
      << std::left << std::setw(12) << std::setprecision(3) << bandwidth
      << std::left << std::setw(12) << std::setprecision(5) << *minmax.first
      << std::left << std::setw(12) << std::setprecision(5) << *minmax.second
      << std::left << std::setw(12) << std::setprecision(5) << average
      << std::endl;
  }
  // Add a blank line
  std::cout << std::endl;


}


int parseUInt(const char *str, unsigned int *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

int parseInt(const char *str, int *output)
{
  char *next;
  *output = strtol(str, &next, 10);
  return !strlen(next);
}

void parseArguments(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
  {
    if (!std::string("--arraysize").compare(argv[i]) ||
        !std::string("-s").compare(argv[i]))
    {
      if (++i >= argc || !parseInt(argv[i], &ARRAY_SIZE) || ARRAY_SIZE <= 0)
      {
        std::cerr << "Invalid array size." << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--numtimes").compare(argv[i]) ||
        !std::string("-n").compare(argv[i]))
    {
      if (++i >= argc || !parseUInt(argv[i], &num_times))
      {
        std::cerr << "Invalid number of times." << std::endl;
        exit(EXIT_FAILURE);
      }
      if (num_times < 2)
      {
        std::cerr << "Number of times must be 2 or more" << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (!std::string("--help").compare(argv[i]) ||
        !std::string("-h").compare(argv[i]))
    {
      std::cout << std::endl;
      std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  -h  --help               Print the message" << std::endl;
      std::cout << "  -s  --arraysize  SIZE    Use SIZE elements in the array" << std::endl;
      std::cout << "  -n  --numtimes   NUM     Run the test NUM times (NUM >= 2)" << std::endl;
      std::cout << std::endl;
      exit(EXIT_SUCCESS);
    }
    else
    {
      std::cerr << "Unrecognized argument '" << argv[i] << "' (try '--help')"
        << std::endl;
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char *argv[])
{
  parseArguments(argc, argv);
  iris::Platform platform;
  try {
    require_iris(platform.init(&argc, &argv, true));
    int count = 0;
    require_iris(iris_device_count(&count));
    for (int i = 0; i < count; ++i) {
      int type = 0;
      char name[256] = {};
      require_iris(iris_device_info(i, iris_type, &type, nullptr));
      require_iris(iris_device_info(i, iris_name, name, nullptr));
      std::cout << "IRIS device " << i << ": " << name << " type=" << type << std::endl;
      if ((type & iris_amd) && selected_device < 0) selected_device = i;
    }
    if (selected_device < 0) throw std::runtime_error("No IRIS AMD GPU available");
    std::cout << "Selected IRIS AMD device " << selected_device << std::endl;
    run<float>();
    run<double>();
    const int errors = platform.error_count();
    require_iris(platform.finalize());
    return errors ? EXIT_FAILURE : EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    platform.finalize();
    return EXIT_FAILURE;
  }
}
