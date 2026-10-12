#include <iris/iris.hpp>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define MAXDISTANCE 200

static void floydWarshallCPUReference(unsigned int* pathDistanceMatrix,
                                      unsigned int* pathMatrix,
                                      unsigned int numNodes)
{
  for (unsigned int k = 0; k < numNodes; ++k) {
    for (unsigned int y = 0; y < numNodes; ++y) {
      const unsigned int yXwidth = y * numNodes;
      for (unsigned int x = 0; x < numNodes; ++x) {
        const unsigned int distanceYtoX = pathDistanceMatrix[yXwidth + x];
        const unsigned int distanceYtoK = pathDistanceMatrix[yXwidth + k];
        const unsigned int distanceKtoX = pathDistanceMatrix[k * numNodes + x];
        const unsigned int indirectDistance = distanceYtoK + distanceKtoX;
        if (indirectDistance < distanceYtoX) {
          pathDistanceMatrix[yXwidth + x] = indirectDistance;
          pathMatrix[yXwidth + x] = k;
        }
      }
    }
  }
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::printf("Usage: %s <number of nodes> <iterations> <block size>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  unsigned int numNodes = std::atoi(argv[1]);
  const unsigned int numIterations = std::atoi(argv[2]);
  unsigned int blockSize = std::atoi(argv[3]);
  if (numNodes % blockSize != 0) {
    numNodes = (numNodes / blockSize + 1) * blockSize;
  }

  const unsigned int matrixElements = numNodes * numNodes;
  const size_t matrixSizeBytes = matrixElements * sizeof(unsigned int);
  unsigned int* pathDistanceMatrix = static_cast<unsigned int*>(std::malloc(matrixSizeBytes));
  unsigned int* pathMatrix = static_cast<unsigned int*>(std::malloc(matrixSizeBytes));
  unsigned int* verificationPathDistanceMatrix = static_cast<unsigned int*>(std::malloc(matrixSizeBytes));
  unsigned int* verificationPathMatrix = static_cast<unsigned int*>(std::malloc(matrixSizeBytes));
  if (!pathDistanceMatrix || !pathMatrix || !verificationPathDistanceMatrix || !verificationPathMatrix) {
    std::fprintf(stderr, "allocation failure\n");
    platform.finalize();
    return 1;
  }

  std::srand(2);
  for (unsigned int i = 0; i < numNodes; i++) {
    for (unsigned int j = 0; j < numNodes; j++) {
      pathDistanceMatrix[i * numNodes + j] = std::rand() % (MAXDISTANCE + 1);
    }
  }
  for (unsigned int i = 0; i < numNodes; ++i) {
    pathDistanceMatrix[i * numNodes + i] = 0;
  }
  for (unsigned int i = 0; i < numNodes; ++i) {
    for (unsigned int j = 0; j < i; ++j) {
      pathMatrix[i * numNodes + j] = i;
      pathMatrix[j * numNodes + i] = j;
    }
    pathMatrix[i * numNodes + i] = i;
  }

  std::memcpy(verificationPathDistanceMatrix, pathDistanceMatrix, matrixSizeBytes);
  std::memcpy(verificationPathMatrix, pathMatrix, matrixSizeBytes);

  unsigned int localThreads[2] = {blockSize, blockSize};
  if (localThreads[0] * localThreads[0] > 256) {
    blockSize = 16;
    localThreads[0] = blockSize;
    localThreads[1] = blockSize;
  }

  iris::DMem pathDistanceBuffer(pathDistanceMatrix, matrixSizeBytes);
  iris::DMem pathBuffer(pathMatrix, matrixSizeBytes);

  const size_t gws[2] = {numNodes, numNodes};
  const size_t lws[2] = {localThreads[0], localThreads[1]};
  double total_time_ns = 0.0;

  for (unsigned int n = 0; n < numIterations; n++) {
    std::memcpy(pathDistanceMatrix, verificationPathDistanceMatrix, matrixSizeBytes);
    std::memcpy(pathMatrix, verificationPathMatrix, matrixSizeBytes);

    const auto start = std::chrono::steady_clock::now();
    for (unsigned int pass = 0; pass < numNodes; pass++) {
      iris::Task task;
      void* params[4] = {&pathDistanceBuffer, &pathBuffer, &numNodes, &pass};
      int pinfo[4] = {iris_rw, iris_w, sizeof(numNodes), sizeof(pass)};
      task.kernel("floydWarshallPass", 2, nullptr, const_cast<size_t*>(gws),
                  const_cast<size_t*>(lws), 4, params, pinfo);
      if (pass + 1 == numNodes) task.flush_out(pathDistanceBuffer);
      task.submit(iris_gpu, nullptr, 1);
    }
    const auto end = std::chrono::steady_clock::now();
    total_time_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  std::printf("Average kernel execution time %f (s)\n", (total_time_ns * 1e-9) / numIterations);

  floydWarshallCPUReference(verificationPathDistanceMatrix, verificationPathMatrix, numNodes);
  if (std::memcmp(pathDistanceMatrix, verificationPathDistanceMatrix, matrixSizeBytes) == 0) {
    std::printf("PASS\n");
  } else {
    std::printf("FAIL\n");
  }

  std::free(pathDistanceMatrix);
  std::free(pathMatrix);
  std::free(verificationPathDistanceMatrix);
  std::free(verificationPathMatrix);
  platform.finalize();
  return 0;
}
