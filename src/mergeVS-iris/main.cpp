#include <iris/iris.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "reference.h"

struct MergeheadResult {
  std::vector<int> block_count;
  std::vector<int> block_offset;
  std::vector<int> column_count;
  std::vector<int> column_index;
  int n_scalar;
  int n_offset;
  int n_index;
  int num_rows;
  int NNZ_V;
  int NNZ_S;
};

static void init_input(int* h_vidx, int* h_sidx, int* h_vcnt, int* h_scnt,
                       int BATCH, int N_HEADS, int NNZ_V, int NNZ_S,
                       int num_rows, int n_sinks = 2, int window_diags = 4)
{
  int eff_v = (n_sinks < NNZ_V) ? n_sinks : NNZ_V;
  int eff_s = (window_diags < NNZ_S) ? window_diags : NNZ_S;
  for (int b = 0; b < BATCH; ++b) {
    for (int h = 0; h < N_HEADS; ++h) {
      int base_v = (b * N_HEADS + h) * NNZ_V;
      int base_s = (b * N_HEADS + h) * NNZ_S;
      for (int i = 0; i < eff_v; ++i) h_vidx[base_v + i] = i;
      for (int i = 0; i < eff_s; ++i) h_sidx[base_s + i] = num_rows - 1 + i;
    }
  }
  for (int h = 0; h < N_HEADS; ++h) {
    h_vcnt[h] = eff_v;
    h_scnt[h] = eff_s;
  }
}

struct Config {
  int batch, n_heads, nnz_v, nnz_s, context_size, block_size;
};

static Config cases[] = {
    {1, 8, 16, 16, 4096, 64},
    {1, 32, 32, 32, 131072, 128},
    {4, 64, 64, 64, 16384, 64},
    {8, 32, 64, 64, 32768, 64},
    {16, 64, 256, 256, 65536, 64},
};

static bool run_test(int repeat, bool causal, const Config& cfg)
{
  const int BATCH = cfg.batch;
  const int N_HEADS = cfg.n_heads;
  const int NNZ_V = cfg.nnz_v;
  const int NNZ_S = cfg.nnz_s;
  const int context_size = cfg.context_size;
  const int block_size_M = cfg.block_size;
  const int block_size_N = cfg.block_size;
  std::printf("batch %4d, #heads %3d, nnz_v %3d, nnz_s %3d, context size %6d, block size %3d, causal %d\n",
              BATCH, N_HEADS, NNZ_V, NNZ_S, context_size, block_size_M, causal);

  const int num_rows = (context_size + block_size_M - 1) / block_size_M;
  const size_t qkv_size = BATCH;
  const size_t vidx_size = static_cast<size_t>(BATCH) * N_HEADS * NNZ_V;
  const size_t sidx_size = static_cast<size_t>(BATCH) * N_HEADS * NNZ_S;
  const size_t idxcnt_size = N_HEADS;

  std::vector<int> h_q(BATCH), h_kv(BATCH);
  std::srand(131072);
  for (int i = 0; i < BATCH; i++)
    h_q[i] = h_kv[i] = (1 + std::rand() % (context_size / block_size_M)) * block_size_M;

  std::vector<int> h_vidx(vidx_size, 0), h_sidx(sidx_size, 0);
  std::vector<int> h_vcnt(idxcnt_size, 0), h_scnt(idxcnt_size, 0);
  init_input(h_vidx.data(), h_sidx.data(), h_vcnt.data(), h_scnt.data(),
             BATCH, N_HEADS, NNZ_V, NNZ_S, num_rows);

  MergeheadResult res;
  res.num_rows = num_rows;
  res.NNZ_V = NNZ_V;
  res.NNZ_S = NNZ_S;
  res.n_scalar = BATCH * N_HEADS * num_rows;
  res.n_offset = BATCH * N_HEADS * num_rows * NNZ_S;
  res.n_index = BATCH * N_HEADS * num_rows * NNZ_V;
  res.block_count.assign(res.n_scalar, 0);
  res.column_count.assign(res.n_scalar, 0);
  res.block_offset.assign(res.n_offset, 0);
  res.column_index.assign(res.n_index, 0);

  std::vector<int> r_col_cnt(res.n_scalar, 0), r_blk_cnt(res.n_scalar, 0);
  std::vector<int> r_col_idx(res.n_index, 0), r_blk_off(res.n_offset, 0);
  reference(h_q.data(), h_kv.data(), h_vidx.data(), h_sidx.data(), h_vcnt.data(),
            h_scnt.data(), r_blk_cnt.data(), r_blk_off.data(), r_col_cnt.data(),
            r_col_idx.data(), BATCH, N_HEADS, num_rows, block_size_M, block_size_N,
            NNZ_V, NNZ_S, causal);

  iris::DMem d_q(h_q.data(), qkv_size * sizeof(int));
  iris::DMem d_kv(h_kv.data(), qkv_size * sizeof(int));
  iris::DMem d_vidx(h_vidx.data(), vidx_size * sizeof(int));
  iris::DMem d_sidx(h_sidx.data(), sidx_size * sizeof(int));
  iris::DMem d_vcnt(h_vcnt.data(), idxcnt_size * sizeof(int));
  iris::DMem d_scnt(h_scnt.data(), idxcnt_size * sizeof(int));
  iris::DMem d_blk_cnt(res.block_count.data(), res.block_count.size() * sizeof(int));
  iris::DMem d_blk_off(res.block_offset.data(), res.block_offset.size() * sizeof(int));
  iris::DMem d_col_cnt(res.column_count.data(), res.column_count.size() * sizeof(int));
  iris::DMem d_col_idx(res.column_index.data(), res.column_index.size() * sizeof(int));

  const long long ll_N_HEADS = N_HEADS;
  const long long ll_N_ROWS = num_rows;
  const long long ll_BM = block_size_M;
  const long long ll_BN = block_size_N;
  long long ll_NNZ_V = NNZ_V;
  long long ll_NNZ_S = NNZ_S;
  const bool c_causal = causal;
  size_t gws[3] = {static_cast<size_t>(N_HEADS * 64), static_cast<size_t>(BATCH),
                   static_cast<size_t>((num_rows + 63) / 64)};
  size_t lws[3] = {64, 1, 1};

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[17] = {&d_q,       &d_kv,      &d_vidx,    &d_sidx,    &d_vcnt,
                        &d_scnt,    &d_blk_cnt, &d_blk_off, &d_col_cnt, &d_col_idx,
                        const_cast<long long*>(&ll_N_HEADS),
                        const_cast<long long*>(&ll_N_ROWS),
                        const_cast<long long*>(&ll_BM),
                        const_cast<long long*>(&ll_BN),
                        &ll_NNZ_V,
                        &ll_NNZ_S,
                        const_cast<bool*>(&c_causal)};
    int pinfo[17] = {iris_r, iris_r, iris_r, iris_r, iris_r, iris_r,
                     iris_rw, iris_rw, iris_rw, iris_rw,
                     sizeof(ll_N_HEADS), sizeof(ll_N_ROWS), sizeof(ll_BM),
                     sizeof(ll_BN), sizeof(ll_NNZ_V), sizeof(ll_NNZ_S), sizeof(c_causal)};
    task.kernel("convert_vertical_slash_indexes_kernel_mergehead", 3, nullptr, gws, lws,
                17, params, pinfo);
    if (i == repeat - 1) {
      task.flush_out(d_blk_cnt);
      task.flush_out(d_blk_off);
      task.flush_out(d_col_cnt);
      task.flush_out(d_col_idx);
    }
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

  d_blk_cnt.fetch(res.block_count.data());
  d_col_cnt.fetch(res.column_count.data());
  d_blk_off.fetch(res.block_offset.data());
  d_col_idx.fetch(res.column_index.data());

  int status = 0;
  status += std::memcmp(res.column_count.data(), r_col_cnt.data(), res.n_scalar * sizeof(int));
  status += std::memcmp(res.block_count.data(), r_blk_cnt.data(), res.n_scalar * sizeof(int));
  status += std::memcmp(res.column_index.data(), r_col_idx.data(), res.n_index * sizeof(int));
  status += std::memcmp(res.block_offset.data(), r_blk_off.data(), res.n_offset * sizeof(int));
  std::printf("%s\n", status ? "FAIL" : "PASS");
  return status == 0;
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
  bool ok = true;
  for (int i = 0; i < 5; i++) {
    ok = run_test(repeat, true, cases[i]) && ok;
    ok = run_test(repeat, false, cases[i]) && ok;
  }
  std::printf("----- warmup done -----\n");
  for (int i = 0; i < 5; i++) {
    ok = run_test(repeat, true, cases[i]) && ok;
    ok = run_test(repeat, false, cases[i]) && ok;
  }
  platform.finalize();
  return ok ? 0 : 1;
}
