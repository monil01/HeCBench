#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define VOTE_DATA_GROUP 4

static void genVoteTestPattern(unsigned int* pattern, int size)
{
  for (int i = 0; i < size / 4; i++) pattern[i] = 0x00000000;
  for (int i = 2 * size / 8; i < 4 * size / 8; i++) {
    pattern[i] = (i & 0x01) ? static_cast<unsigned int>(i) : 0U;
  }
  for (int i = 2 * size / 4; i < 3 * size / 4; i++) {
    pattern[i] = (i & 0x01) ? 0U : static_cast<unsigned int>(i);
  }
  for (int i = 3 * size / 4; i < 4 * size / 4; i++) {
    pattern[i] = 0xffffffffU;
  }
}

static int checkErrors1(unsigned int* h_result, int start, int end,
                        const char* voteType)
{
  int sum = 0;
  for (int i = start; i < end; i++) sum += h_result[i];
  if (sum > 0) {
    std::printf("\t<%s>[%d - %d] = ", voteType, start, end - 1);
    for (int i = start; i < end; i++) std::printf("%d", h_result[i]);
    std::printf("%d values FAILED\n", sum);
  }
  return (sum > 0);
}

static int checkErrors2(unsigned int* h_result, int start, int end,
                        int warp_size, const char* voteType)
{
  int sum = 0;
  for (int i = start; i < end; i++) sum += h_result[i];
  if (sum != warp_size) {
    std::printf("\t<%s>[%d - %d] = ", voteType, start, end - 1);
    for (int i = start; i < end; i++) std::printf("%d", h_result[i]);
    std::printf(" - FAILED\n");
  }
  return (sum != warp_size);
}

static int checkResultsVoteAnyKernel1(unsigned int* h_result, int warp_size)
{
  int error_count = 0;
  error_count += checkErrors1(h_result, 0, VOTE_DATA_GROUP * warp_size / 4,
                              "Vote.Any");
  error_count += checkErrors2(h_result, VOTE_DATA_GROUP * warp_size / 4,
                              2 * VOTE_DATA_GROUP * warp_size / 4, warp_size,
                              "Vote.Any");
  error_count += checkErrors2(h_result, 2 * VOTE_DATA_GROUP * warp_size / 4,
                              3 * VOTE_DATA_GROUP * warp_size / 4, warp_size,
                              "Vote.Any");
  error_count += checkErrors2(h_result, 3 * VOTE_DATA_GROUP * warp_size / 4,
                              4 * VOTE_DATA_GROUP * warp_size / 4, warp_size,
                              "Vote.Any");
  std::printf((error_count == 0) ? "\tOK\n" : "\tERROR\n");
  return error_count;
}

static int checkResultsVoteAllKernel2(unsigned int* h_result, int warp_size)
{
  int error_count = 0;
  error_count += checkErrors1(h_result, 0, VOTE_DATA_GROUP * warp_size / 4,
                              "Vote.All");
  error_count += checkErrors1(h_result, VOTE_DATA_GROUP * warp_size / 4,
                              2 * VOTE_DATA_GROUP * warp_size / 4,
                              "Vote.All");
  error_count += checkErrors1(h_result, 2 * VOTE_DATA_GROUP * warp_size / 4,
                              3 * VOTE_DATA_GROUP * warp_size / 4,
                              "Vote.All");
  error_count += checkErrors2(h_result, 3 * VOTE_DATA_GROUP * warp_size / 4,
                              4 * VOTE_DATA_GROUP * warp_size / 4, warp_size,
                              "Vote.All");
  std::printf((error_count == 0) ? "\tOK\n" : "\tERROR\n");
  return error_count;
}

static int checkResultsVoteAnyKernel3(bool* hinfo, int size)
{
  int error_count = 0;
  for (int i = 0; i < size * 3; i++) {
    switch (i % 3) {
      case 0:
        if (hinfo[i] != (i >= size * 1)) error_count++;
        break;
      case 1:
        if (hinfo[i] != (i >= size * 3 / 2)) error_count++;
        break;
      case 2:
        if (hinfo[i] != (i >= size * 2)) error_count++;
        break;
    }
  }
  std::printf((error_count == 0) ? "\tOK\n" : "\tERROR\n");
  return error_count;
}

static void run_kernel(const char* name, int nparams, void** params, int* pinfo,
                       size_t gws, size_t lws, iris::DMem* flush)
{
  iris::Task task;
  size_t global[1] = {gws};
  size_t local[1] = {lws};
  task.kernel(name, 1, nullptr, global, local, nparams, params, pinfo);
  if (flush) task.flush_out(*flush);
  task.submit(iris_gpu, nullptr, 1);
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = std::atoi(argv[1]);
  const int warp_size = 64;
  const int vote_items = VOTE_DATA_GROUP * warp_size;
  const size_t vote_bytes = vote_items * sizeof(unsigned int);
  unsigned int* h_input = static_cast<unsigned int*>(std::malloc(vote_bytes));
  unsigned int* h_result = static_cast<unsigned int*>(std::malloc(vote_bytes));
  if (!h_input || !h_result) {
    std::fprintf(stderr, "host allocation failed\n");
    return 1;
  }
  genVoteTestPattern(h_input, vote_items);

  iris::DMem d_input(h_input, vote_bytes);
  iris::DMem d_result(h_result, vote_bytes);
  int error_count[3] = {0, 0, 0};

  std::printf("\tRunning <<Vote.Any>> kernel1 ...\n");
  void* params1[3] = {&d_input, &d_result, const_cast<int*>(&repeat)};
  int pinfo1[3] = {iris_r, iris_w, sizeof(repeat)};
  run_kernel("VoteAnyKernel1", 3, params1, pinfo1, vote_items, vote_items, nullptr);
  auto start = std::chrono::steady_clock::now();
  run_kernel("VoteAnyKernel1", 3, params1, pinfo1, vote_items, vote_items, &d_result);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("\tkernel execution time: %f (s)\n", time * 1e-9f);
  d_result.fetch(h_result);
  error_count[0] += checkResultsVoteAnyKernel1(h_result, warp_size);

  std::printf("\tRunning <<Vote.All>> kernel2 ...\n");
  void* params2[3] = {&d_input, &d_result, const_cast<int*>(&repeat)};
  int pinfo2[3] = {iris_r, iris_w, sizeof(repeat)};
  run_kernel("VoteAllKernel2", 3, params2, pinfo2, vote_items, vote_items, nullptr);
  start = std::chrono::steady_clock::now();
  run_kernel("VoteAllKernel2", 3, params2, pinfo2, vote_items, vote_items, &d_result);
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("\tkernel execution time: %f (s)\n", time * 1e-9f);
  d_result.fetch(h_result);
  error_count[1] += checkResultsVoteAllKernel2(h_result, warp_size);

  const int info_items = warp_size * 3 * 3;
  const size_t info_bytes = info_items * sizeof(bool);
  bool* hinfo = static_cast<bool*>(std::calloc(info_items, sizeof(bool)));
  if (!hinfo) {
    std::fprintf(stderr, "host info allocation failed\n");
    return 1;
  }
  iris::DMem dinfo(hinfo, info_bytes);

  void* params3[3] = {&dinfo, const_cast<int*>(&warp_size), const_cast<int*>(&repeat)};
  int pinfo3[3] = {iris_rw, sizeof(warp_size), sizeof(repeat)};
  run_kernel("VoteAnyKernel3", 3, params3, pinfo3, warp_size * 3, warp_size * 3, nullptr);

  void* reset_params[2] = {&dinfo, const_cast<int*>(&info_items)};
  int reset_pinfo[2] = {iris_w, sizeof(info_items)};
  run_kernel("VoteResetInfo", 2, reset_params, reset_pinfo, info_items, 256, nullptr);

  std::printf("\tRunning <<Vote.Any>> kernel3 ...\n");
  start = std::chrono::steady_clock::now();
  run_kernel("VoteAnyKernel3", 3, params3, pinfo3, warp_size * 3, warp_size * 3, &dinfo);
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("\tkernel execution time: %f (s)\n", time * 1e-9f);
  dinfo.fetch(hinfo);
  error_count[2] = checkResultsVoteAnyKernel3(hinfo, warp_size * 3);

  std::free(h_input);
  std::free(h_result);
  std::free(hinfo);
  platform.finalize();
  return (error_count[0] == 0 && error_count[1] == 0 && error_count[2] == 0)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
