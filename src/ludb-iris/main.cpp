#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define N 48
#define BATCH_SIZE 10000

#define DATA_TYPE float
#define MAX_ERROR 1e-6f

static void* xmalloc(size_t size)
{
  void* ptr = std::malloc(size);
  if (ptr == nullptr) {
    std::printf("> ERROR: malloc for size %zu failed..\n", size);
    std::exit(EXIT_FAILURE);
  }
  std::memset(ptr, 0, size);
  return ptr;
}

static void initIdentityMatrix(DATA_TYPE* mat)
{
  std::memset(mat, 0, N * N * sizeof(DATA_TYPE));
  for (int i = 0; i < N; i++) {
    mat[(i * N) + i] = 1.0f;
  }
}

static void initZeroMatrix(DATA_TYPE* mat)
{
  std::memset(mat, 0, N * N * sizeof(DATA_TYPE));
}

static void initRandomMatrix(DATA_TYPE* mat)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      mat[(j * N) + i] = 1.0f + (static_cast<DATA_TYPE>(std::rand()) / static_cast<DATA_TYPE>(RAND_MAX));
    }
  }

  for (int i = 0; i < N; i++) {
    mat[(i * N) + i] += static_cast<DATA_TYPE>(N);
  }
}

static void matrixMultiply(DATA_TYPE* res, DATA_TYPE* mat1, DATA_TYPE* mat2)
{
  initZeroMatrix(res);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        res[(j * N) + i] += mat1[(k * N) + i] * mat2[(j * N) + k];
      }
    }
  }
}

static bool checkRelativeError(DATA_TYPE* mat1, DATA_TYPE* mat2, DATA_TYPE maxError)
{
  DATA_TYPE relMaxError = 0.0f;
  for (int i = 0; i < N * N; i++) {
    DATA_TYPE refNorm = std::fabs(mat1[i]);
    DATA_TYPE err = std::fabs(mat1[i] - mat2[i]);
    if (refNorm != 0.0f && err > 0.0f) {
      DATA_TYPE relError = err / refNorm;
      relMaxError = relMaxError > relError ? relMaxError : relError;
    }
    if (relMaxError > maxError) return false;
  }
  return true;
}

static void getLUdecoded(DATA_TYPE* mat, DATA_TYPE* L, DATA_TYPE* U)
{
  initIdentityMatrix(L);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < i; j++) {
      L[(j * N) + i] = mat[(j * N) + i];
    }
  }

  initZeroMatrix(U);
  for (int i = 0; i < N; i++) {
    for (int j = i; j < N; j++) {
      U[(j * N) + i] = mat[(j * N) + i];
    }
  }
}

static void getPmatFromPivot(DATA_TYPE* Pmat, int* P)
{
  int pivot[N];
  for (int i = 0; i < N; i++) {
    P[i]--;
  }

  for (int k = 0; k < N; k++) {
    pivot[k] = k;
  }

  for (int k = 0; k < N; k++) {
    int q = P[k];
    int s = pivot[k];
    int t = pivot[q];
    pivot[k] = t;
    pivot[q] = s;
  }

  initZeroMatrix(Pmat);
  for (int i = 0; i < N; i++) {
    int j = pivot[i];
    Pmat[(j * N) + i] = 1.0f;
  }
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);

  std::printf("> initializing..\n");
  std::printf("> using SINGLE precision..\n");
  std::printf("> pivot ENABLED..\n");

  const size_t matSize = N * N * sizeof(DATA_TYPE);
  DATA_TYPE* h_AarrayInput = static_cast<DATA_TYPE*>(xmalloc(BATCH_SIZE * matSize));
  DATA_TYPE* h_AarrayOutput = static_cast<DATA_TYPE*>(xmalloc(BATCH_SIZE * matSize));
  int* h_pivotArray = static_cast<int*>(xmalloc(N * BATCH_SIZE * sizeof(int)));
  int* h_infoArray = static_cast<int*>(xmalloc(BATCH_SIZE * sizeof(int)));

  std::srand(12345);
  std::printf("> generating random matrices..\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    initRandomMatrix(h_AarrayInput + (i * N * N));
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  iris::DMem mem_A(h_AarrayOutput, BATCH_SIZE * matSize);
  iris::DMem mem_pivot(h_pivotArray, N * BATCH_SIZE * sizeof(int));
  iris::DMem mem_info(h_infoArray, BATCH_SIZE * sizeof(int));

  long time = 0;
  std::printf("> performing batched LU decomposition..\n");
  for (int i = 0; i <= repeat; i++) {
    std::memcpy(h_AarrayOutput, h_AarrayInput, BATCH_SIZE * matSize);
    std::memset(h_pivotArray, 0, N * BATCH_SIZE * sizeof(int));
    std::memset(h_infoArray, 0, BATCH_SIZE * sizeof(int));
    mem_A.update(h_AarrayOutput);
    mem_pivot.update(h_pivotArray);
    mem_info.update(h_infoArray);

    size_t lws[1] = {256};
    size_t gws[1] = {((static_cast<size_t>(BATCH_SIZE) + lws[0] - 1) / lws[0]) * lws[0]};
    void* params[4] = {&mem_A, &mem_pivot, &mem_info};
    int batch_size = BATCH_SIZE;
    params[3] = &batch_size;
    int pinfo[4] = {iris_rw, iris_w, iris_w, sizeof(batch_size)};

    auto start = std::chrono::steady_clock::now();
    iris::Task task;
    task.kernel("ludb_getrf_batched", 1, nullptr, gws, lws, 4, params, pinfo);
    task.flush_out(mem_A);
    task.flush_out(mem_pivot);
    task.flush_out(mem_info);
    task.submit(iris_gpu, nullptr, 1);
    auto end = std::chrono::steady_clock::now();
    if (i != 0) {
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
  }

  mem_A.fetch(h_AarrayOutput);
  mem_info.fetch(h_infoArray);
  mem_pivot.fetch(h_pivotArray);
  platform.finalize();

  std::printf("Average kernel execution time : %f (us)\n", (time * 1e-3f) / repeat);

  int err_count = 0;
  std::printf("> verifying the result..\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    if (h_infoArray[i] == 0) {
      DATA_TYPE* A = h_AarrayInput + (i * N * N);
      DATA_TYPE* LU = h_AarrayOutput + (i * N * N);
      DATA_TYPE L[N * N];
      DATA_TYPE U[N * N];
      getLUdecoded(LU, L, U);

      int* P = h_pivotArray + (i * N);
      DATA_TYPE Pmat[N * N];
      getPmatFromPivot(Pmat, P);

      DATA_TYPE PxA[N * N];
      DATA_TYPE LxU[N * N];
      matrixMultiply(PxA, Pmat, A);
      matrixMultiply(LxU, L, U);

      if (!checkRelativeError(PxA, LxU, static_cast<DATA_TYPE>(MAX_ERROR))) {
        std::printf("> ERROR: accuracy check failed for matrix number %05d..\n", i + 1);
        err_count++;
      }
    } else if (h_infoArray[i] > 0) {
      std::printf("> execution for matrix %05d is successful, but U is singular and U(%d,%d) = 0..\n",
                  i + 1, h_infoArray[i] - 1, h_infoArray[i] - 1);
    } else {
      std::printf("> ERROR: matrix %05d have an illegal value at index %d = %f..\n",
                  i + 1, -h_infoArray[i], *(h_AarrayInput + (i * N * N) + (-h_infoArray[i])));
    }
  }

  std::free(h_infoArray);
  std::free(h_pivotArray);
  std::free(h_AarrayOutput);
  std::free(h_AarrayInput);

  if (err_count > 0) {
    std::printf("> TEST FAILED for %d matrices, with precision: %g\n", err_count, static_cast<double>(MAX_ERROR));
    return EXIT_FAILURE;
  }

  std::printf("> TEST SUCCESSFUL, with precision: %g\n", static_cast<double>(MAX_ERROR));
  return EXIT_SUCCESS;
}
