/** Modifed version of knn-CUDA from https://github.com/vincentfpgarcia/kNN-CUDA
 * The modifications are
 *      removed texture memory usage
 *      removed split query KNN computation
 *      added feature extraction with bilinear interpolation
 *
 * Last modified by Christopher B. Choy <chrischoy@ai.stanford.edu> 12/23/2016
 */

// Includes
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "dmem_runtime.hpp"

// Constants used by the program
#define BLOCK_DIM 16

/**
  * Computes the distance between two matrix A (reference points) and
  * B (query points) containing respectively wA and wB points.
  *
  * @param A     pointer on the matrix A
  * @param wA    width of the matrix A = number of points in A
  * @param B     pointer on the matrix B
  * @param wB    width of the matrix B = number of points in B
  * @param dim   dimension of points = height of matrices A and B
  * @param AB    pointer on the matrix containing the wA*wB distances computed
  */
void knn_parallel(float *ref_host, int ref_width, float *query_host,
              int query_width, int height, int k, float *dist_host,
              int *ind_host) {

  unsigned int size_of_float = sizeof(float);
  unsigned int size_of_int = sizeof(int);

  // Variables
  float *query_dev;
  float *ref_dev;
  float *dist_dev;
  int *ind_dev;

  // Allocation of global memory for query points and for distances, CUDA_CHECK
  iris_alloc((void **)&query_dev, query_width * height * size_of_float);
  iris_alloc((void **)&dist_dev, query_width * ref_width * size_of_float);

  // Allocation of global memory for indexes CUDA_CHECK
  iris_alloc((void **)&ind_dev, query_width * k * size_of_int);

  // Allocation of global memory CUDA_CHECK
  iris_alloc((void **)&ref_dev, ref_width * height * size_of_float);

  iris_copy(ref_dev, ref_host, ref_width * height * size_of_float,
             iris_h2d);

  // Copy of part of query actually being treated
  iris_copy(query_dev, query_host, query_width * height * size_of_float,
             iris_h2d);

  // Grids ans threads
  dim3 g_16x16((query_width + 15) / 16, (ref_width + 15) / 16, 1);
  dim3 t_16x16(16, 16, 1);
  //
  dim3 g_256x1((query_width + 255) / 256, 1, 1);
  dim3 t_256x1(256, 1, 1);

  dim3 g_k_16x16((query_width + 15) / 16, (k + 15) / 16, 1);
  dim3 t_k_16x16(16, 16, 1);

  // Kernel 1: Compute all the distances
  iris_launch("cuComputeDistanceGlobal", dim3(g_16x16), dim3(t_16x16), ref_dev, ref_width, query_dev,
                                                query_width, height, dist_dev);

#ifdef DEBUG
  iris_copy(dist_host, dist_dev, query_width * ref_width * size_of_float,
             iris_d2h);

  for (int i = 0; i < query_width * ref_width; i++)
    printf("k1 dist: %d %f\n", i, dist_host[i]);
#endif

  // Kernel 2: Sort each column
  iris_launch("cuInsertionSort", dim3(g_256x1), dim3(t_256x1), dist_dev, ind_dev, query_width,
                                        ref_width, k);

#ifdef DEBUG
  iris_copy(dist_host, dist_dev, query_width * ref_width * size_of_float,
             iris_d2h);

  for (int i = 0; i < query_width * ref_width; i++)
    printf("k2 dist: %d %f\n", i, dist_host[i]);

  iris_copy(ind_host, ind_dev, query_width * k * size_of_int,
             iris_d2h);
  for (int i = 0; i < query_width * k; i++)
    printf("k2 index: %d %d\n", i, ind_host[i]);
#endif

  // Kernel 3: Compute square root of k first elements
  iris_launch("cuParallelSqrt", dim3(g_k_16x16), dim3(t_k_16x16), dist_dev, query_width, k);
  iris_sync();
  // Memory copy of output from device to host
  iris_copy(dist_host, dist_dev, query_width * k * size_of_float,
             iris_d2h);

  iris_copy(ind_host, ind_dev, query_width * k * size_of_int,
             iris_d2h);

  // Free memory
  iris_free(ref_dev);
  iris_free(ind_dev);
  iris_free(query_dev);
  iris_free(dist_dev);
}

float compute_distance(const float *ref, int ref_nb, const float *query,
                       int query_nb, int dim, int ref_index, int query_index) {
  float sum = 0.f;
  for (int d = 0; d < dim; ++d) {
    const float diff =
        ref[d * ref_nb + ref_index] - query[d * query_nb + query_index];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

void modified_insertion_sort(float *dist, int *index, int length, int k) {

  // Initialise the first index
  index[0] = 0;

  // Go through all points
  for (int i = 1; i < length; ++i) {

    // Store current distance and associated index
    float curr_dist = dist[i];
    int curr_index = i;

    // Skip the current value if its index is >= k and if it's higher the k-th
    // slready sorted mallest value
    if (i >= k && curr_dist >= dist[k - 1]) {
      continue;
    }

    // Shift values (and indexes) higher that the current distance to the right
    int j = i < k - 1 ? i : k-1;
    while (j > 0 && dist[j - 1] > curr_dist) {
      dist[j] = dist[j - 1];
      index[j] = index[j - 1];
      --j;
    }

    // Write the current distance and index at their position
    dist[j] = curr_dist;
    index[j] = curr_index;
  }
}

bool knn_c(const float *ref, int ref_nb, const float *query, int query_nb,
           int dim, int k, float *knn_dist, int *knn_index) {
  // Allocate local array to store all the distances / indexes for a given query
  // point
  float *dist = (float *)malloc(ref_nb * sizeof(float));
  int *index = (int *)malloc(ref_nb * sizeof(int));

  // Allocation checks
  if (!dist || !index) {
    printf("Memory allocation error\n");
    free(dist);
    free(index);
    return false;
  }

  // Process one query point at the time
  for (int i = 0; i < query_nb; ++i) {

    // Compute all distances / indexes
    for (int j = 0; j < ref_nb; ++j) {
      dist[j] = compute_distance(ref, ref_nb, query, query_nb, dim, j, i);
      index[j] = j;
    }

    // Sort distances / indexes
    modified_insertion_sort(dist, index, ref_nb, k);

    // Copy k smallest distances and their associated index
    for (int j = 0; j < k; ++j) {
      knn_dist[j * query_nb + i] = dist[j];
      knn_index[j * query_nb + i] = index[j];
    }
  }

  // Memory clean-up
  free(dist);
  free(index);
  return true;
}

/**
  * Example of use of kNN search CUDA.
  */
int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int iterations = atoi(argv[1]);
  
  // Variables and parameters
  float *ref;          // Pointer to reference point array
  float *query;        // Pointer to query point array
  float *dist;         // Pointer to distance array
  int *ind;            // Pointer to index array
  int ref_nb = 4096;   // Reference point number, max=65535
  int query_nb = 4096; // Query point number,     max=65535
  int dim = 68;        // Dimension of points
  int k = 20;          // Nearest neighbors to consider
  int c_iterations = 1;
  int i;
  const float precision = 0.001f; // distance error max
  int nb_correct_precisions = 0;
  int nb_correct_indexes = 0;
  // Memory allocation
  ref = (float *)malloc(ref_nb * dim * sizeof(float));
  query = (float *)malloc(query_nb * dim * sizeof(float));
  dist = (float *)malloc(query_nb * k * sizeof(float));
  ind = (int *)malloc(query_nb * k * sizeof(float));

  // Init
  srand(2);
  for (i = 0; i < ref_nb * dim; i++)
    ref[i] = (float)rand() / (float)RAND_MAX;
  for (i = 0; i < query_nb * dim; i++)
    query[i] = (float)rand() / (float)RAND_MAX;

  // Display informations
  printf("Number of reference points      : %6d\n", ref_nb);
  printf("Number of query points          : %6d\n", query_nb);
  printf("Dimension of points             : %4d\n", dim);
  printf("Number of neighbors to consider : %4d\n", k);
  printf("Processing kNN search           :\n");

  float *knn_dist = (float *)malloc(query_nb * k * sizeof(float));
  int *knn_index = (int *)malloc(query_nb * k * sizeof(int));
  printf("Ground truth computation in progress...\n\n");
  if (!knn_c(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index)) {
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);
    return EXIT_FAILURE;
  }

  printf("On CPU: \n");
  auto start = std::chrono::steady_clock::now();
  for (i = 0; i < c_iterations; i++) {
    knn_c(ref, ref_nb, query, query_nb, dim, k, dist, ind);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf(" done in %f s for %d iterations (%f s by iteration)\n", time * 1e-9,
         c_iterations, time * 1e-9 / (c_iterations));

  printf("on GPU: \n");
  start = std::chrono::steady_clock::now();
  for (i = 0; i < iterations; i++) {
    knn_parallel(ref, ref_nb, query, query_nb, dim, k, dist, ind);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf(" done in %f s for %d iterations (%f s by iteration)\n", time * 1e-9,
         c_iterations, time * 1e-9 / (c_iterations));

  for (int i = 0; i < query_nb * k; ++i) {
    if (fabs(dist[i] - knn_dist[i]) <= precision) {
      nb_correct_precisions++;
    }
    if (ind[i] == knn_index[i]) {
      nb_correct_indexes++;
    }
  }

  float precision_accuracy = nb_correct_precisions / ((float)query_nb * k);
  float index_accuracy = nb_correct_indexes / ((float)query_nb * k);
  printf("Precision accuracy %f\nIndex accuracy %f\n", precision_accuracy, index_accuracy);
  printf("%s\n", (precision_accuracy == 1.f) ? "PASS" : "FAIL");

  free(ind);
  free(dist);
  free(query);
  free(ref);
  // Finalize before C++ static runtime history/pool destructors execute.
  iris_finish();
}
