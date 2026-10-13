// Copyright (c) 2015 Matthias Noack (ma.noack.pr@gmail.com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "utils.hpp"
#include <iris/iris.hpp>
inline void iris_check(int rc) { if (rc != IRIS_SUCCESS) { std::cerr << "IRIS error " << rc << std::endl; exit(1); } }

long benchmark(
    complex_t *sigma_in,
    complex_t *sigma_out,
    complex_t *hamiltonian,
    size_t size_sigma,
    size_t size_hamiltonian,
    complex_t *sigma_reference,
    complex_t *sigma_reference_transformed ,
    const int dim,
    const int num, // global_work_size
    const int kernel_id,
    size_t vec_length,
    decltype(&transform_matrices_aos_to_aosoa) transformation_sigma,
    bool scale_hamiltonian,
    decltype(&transform_matrix_aos_to_soa) transformation_hamiltonian)
{
  initialise_hamiltonian(hamiltonian, dim);

  if (scale_hamiltonian)
    transform_matrix_scale_aos(hamiltonian, dim); // pre-scale hamiltonian

  if (transformation_hamiltonian)
    transformation_hamiltonian(hamiltonian, dim);

  initialise_sigma(sigma_in, sigma_out, dim, num);

  std::memcpy(sigma_reference_transformed, sigma_reference, size_sigma * sizeof(complex_t));

  // transform memory layout if a transformation is specified
  if (transformation_sigma) {
    // transform reference for comparison
    transformation_sigma(sigma_reference_transformed, dim, num, vec_length);

    // transform sigma
    transformation_sigma(sigma_in, dim, num, vec_length);
  }

  // extract the real and imag data
  real_2_t* ham = allocate_aligned<real_2_t>(size_hamiltonian);
  real_2_t* sin = allocate_aligned<real_2_t>(size_sigma);
  real_2_t* sout = allocate_aligned<real_2_t>(size_sigma);

  for (size_t i = 0; i < size_hamiltonian; i++) {
    ham[i].x = hamiltonian[i].real();
    ham[i].y = hamiltonian[i].imag();
  }

  for (size_t i = 0; i < size_sigma; i++) {
    sin[i].x = sigma_in[i].real();
    sin[i].y = sigma_in[i].imag();
  }

  for (size_t i = 0; i < size_sigma; i++) {
    sout[i].x = sigma_out[i].real();
    sout[i].y = sigma_out[i].imag();
  }

  iris::DMem d_hamiltonian(ham, sizeof(real_2_t) * size_hamiltonian);
  iris::DMem d_sigma_in(sin, sizeof(real_2_t) * size_sigma);
  iris::DMem d_sigma_out(sout, sizeof(real_2_t) * size_sigma);

  long total_time = 0;

  // benchmark loop
  for (size_t i = 0; i < NUM_ITERATIONS; ++i) {

    iris_check(d_sigma_out.update(sout));
    auto start = std::chrono::steady_clock::now();
    static const char *kernel_names[] = { "comm_empty",
      "comm_init",
      "comm_refactor",
      "comm_refactor_direct_store",
      "comm_aosoa_naive",
      "comm_aosoa_naive_constants",
      "comm_aosoa_naive_constants_perm",
      "comm_aosoa_naive_direct",
      "comm_aosoa_naive_constants_direct",
      "comm_aosoa_naive_constants_direct_perm",
      "comm_aosoa",
      "comm_aosoa_constants",
      "comm_aosoa_constants_perm",
      "comm_aosoa_direct",
      "comm_aosoa_constants_direct",
      "comm_aosoa_constants_direct_perm",
      "comm_manual_aosoa",
      "comm_manual_aosoa_constants",
      "comm_manual_aosoa_constants_perm",
      "comm_manual_aosoa_constants_perm_prefetch",
      "comm_manual_aosoa_direct",
      "comm_manual_aosoa_constants_direct",
      "comm_manual_aosoa_constants_direct_prefetch",
      "comm_manual_aosoa_constants_direct_perm",
      "final_gpu_kernel" };
    iris::Task task;
    void *args[] = { &d_sigma_in, &d_sigma_out, &d_hamiltonian, const_cast<int*>(&dim) };
    int info[] = { iris_r, iris_rw, iris_r, sizeof(dim) };
    size_t gws[] = { size_t(num / (VEC_LENGTH_AUTO * PACKAGES_PER_WG)) * VEC_LENGTH_AUTO * PACKAGES_PER_WG, 1 };
    size_t lws[] = { VEC_LENGTH_AUTO * PACKAGES_PER_WG, 1 };
    int dimensions = 1;
    if (kernel_id >= 10 && kernel_id <= 15) {
      dimensions = 2;
      gws[0] = VEC_LENGTH_AUTO;
      gws[1] = size_t(num / VEC_LENGTH_AUTO / PACKAGES_PER_WG) * PACKAGES_PER_WG;
      lws[0] = VEC_LENGTH_AUTO; lws[1] = PACKAGES_PER_WG;
    } else if (kernel_id >= 16 && kernel_id <= 23) {
      gws[0] = size_t(num / (VEC_LENGTH * VEC_LENGTH)) * VEC_LENGTH;
      lws[0] = VEC_LENGTH;
    } else if (kernel_id == 24) {
      dimensions = 2;
      lws[0] = (dim * dim + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
      lws[1] = NUM_SUB_GROUPS;
      gws[0] = size_t(num / (NUM_SUB_GROUPS * CHUNK_SIZE)) * lws[0];
      gws[1] = lws[1];
      args[3] = const_cast<int*>(&num); info[3] = sizeof(num);
    }
    const bool has_scalar = (kernel_id >= 1 && kernel_id <= 5) || kernel_id == 7 ||
                            kernel_id == 10 || kernel_id == 13 || kernel_id == 16 ||
                            kernel_id == 20 || kernel_id == 24;
    iris_check(task.kernel(kernel_names[kernel_id], dimensions, nullptr, gws, lws,
                           has_scalar ? 4 : 3, args, info));
    if (i + 1 == NUM_ITERATIONS) iris_check(task.flush_out(d_sigma_out));
    iris_check(task.submit(iris_gpu, nullptr, true));
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }

  std::cout << "Total execution time of kernel "
            << look_up(kernel_id)  << " : " << total_time * 1e-9 << " (s)" << std::endl;

  real_t deviation = 0;

  if (kernel_id > 0)  {



    for (size_t i = 0; i < size_sigma; i++) {
      sigma_out[i] = {sout[i].x, sout[i].y};
    }

    // measure the differences between the CPU and GPU results
    deviation = compare_matrices(sigma_out, sigma_reference_transformed, dim, num);

    std::cout << "Deviation of kernel " << look_up(kernel_id) << ": " << deviation;

  } else {
    // the deviation of an empty kernel does not make sense
    std::cout << "Deviation of kernel " << look_up(kernel_id) << "N/A";
  }

  std::cout << std::endl << std::endl;




  free(sin);
  free(sout);
  free(ham);

  return total_time;
}

int main(int argc, char* argv[])
{
  iris::Platform platform;
  iris_check(platform.init(&argc, &argv, 1));
  // debugging
  print_compile_config(std::cout);

  // constants
  const size_t dim = DIM;
  const size_t num = NUM;

  // allocate host memory
  size_t size_hamiltonian = dim * dim;
  size_t size_sigma = size_hamiltonian * num;
  size_t size_sigma_byte = sizeof(complex_t) * size_sigma;

  complex_t* hamiltonian = allocate_aligned<complex_t>(size_hamiltonian);
  complex_t* sigma_in = allocate_aligned<complex_t>(size_sigma);
  complex_t* sigma_out = allocate_aligned<complex_t>(size_sigma);
  complex_t* sigma_reference = allocate_aligned<complex_t>(size_sigma);
  complex_t* sigma_reference_transformed = allocate_aligned<complex_t>(size_sigma);

  // perform reference computation for correctness analysis
  initialise_hamiltonian(hamiltonian, dim);
  initialise_sigma(sigma_in, sigma_out, dim, num);
  commutator_reference(sigma_in, sigma_out, hamiltonian, dim, num);

  // copy reference results
  std::memcpy(sigma_reference, sigma_out, size_sigma_byte);

  // total kernel time for all benchmarks
  long ktime = 0;

  // The macro "BENCHMARK(...)" is defined in utils.hpp
  ktime += BENCHMARK(0, VEC_LENGTH, NO_TRANSFORM, NO_SCALE_HAMILT, NO_TRANSFORM);

  ktime += BENCHMARK(1, VEC_LENGTH, NO_TRANSFORM, NO_SCALE_HAMILT, NO_TRANSFORM);

  ktime += BENCHMARK(2, VEC_LENGTH, NO_TRANSFORM, NO_SCALE_HAMILT, NO_TRANSFORM);

  ktime += BENCHMARK(3, VEC_LENGTH, NO_TRANSFORM, SCALE_HAMILT, NO_TRANSFORM);

  ktime += BENCHMARK(4, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(5, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(6, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(7, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(8, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(9, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(10, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(11, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(12, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(13, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(14, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(15, VEC_LENGTH_AUTO, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(16, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(17, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(18, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(19, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(20, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(21, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(22, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(23, VEC_LENGTH, &transform_matrices_aos_to_aosoa, SCALE_HAMILT, &transform_matrix_aos_to_soa);

  ktime += BENCHMARK(24, 2, NO_TRANSFORM, SCALE_HAMILT, NO_TRANSFORM);

  printf("Total kernel time for all benchmarks %lf (s)\n", ktime * 1e-9);

  free(hamiltonian);
  free(sigma_in);
  free(sigma_out);
  free(sigma_reference);
  free(sigma_reference_transformed);

  iris_check(platform.finalize());
  return 0;
}
