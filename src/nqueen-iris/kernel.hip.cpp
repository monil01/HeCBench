#include <hip/hip_runtime.h>
#include <stdint.h>

#define _EMPTY_ -1

typedef struct queen_root {
  unsigned int control;
  int8_t board[12];
} QueenRoot;

__device__ bool queens_stillLegal(const char* board, const int r) {
  bool safe = true;
  for (int i = 0; i < r; ++i)
    if (board[i] == board[r]) safe = false;
  int ld = board[r];
  int rd = board[r];
  for (int i = r - 1; i >= 0; --i) {
    --ld;
    ++rd;
    if (board[i] == ld || board[i] == rd) safe = false;
  }
  return safe;
}

extern "C" __global__ void BP_queens_root_dfs(
    int N, unsigned int nPreFixos, int depthPreFixos,
    const QueenRoot* __restrict__ root_prefixes,
    unsigned long long* __restrict__ vector_of_tree_size,
    unsigned long long* __restrict__ sols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nPreFixos) {
    unsigned int flag = 0;
    unsigned int bit_test = 0;
    char vertice[20];
    int N_l = N;
    int i, depth;
    unsigned long long qtd_solutions_thread = 0ULL;
    int depthGlobal = depthPreFixos;
    unsigned long long tree_size = 0ULL;

    for (i = 0; i < N_l; ++i) vertice[i] = _EMPTY_;
    flag = root_prefixes[idx].control;
    for (i = 0; i < depthGlobal; ++i) vertice[i] = root_prefixes[idx].board[i];
    depth = depthGlobal;

    do {
      vertice[depth]++;
      bit_test = 0;
      bit_test |= (1 << vertice[depth]);
      if (vertice[depth] == N_l) {
        vertice[depth] = _EMPTY_;
      } else if (!(flag & bit_test) && queens_stillLegal(vertice, depth)) {
        ++tree_size;
        flag |= (1ULL << vertice[depth]);
        depth++;
        if (depth == N_l) {
          ++qtd_solutions_thread;
        } else {
          continue;
        }
      } else {
        continue;
      }
      depth--;
      flag &= ~(1ULL << vertice[depth]);
    } while (depth >= depthGlobal);

    sols[idx] = qtd_solutions_thread;
    vector_of_tree_size[idx] = tree_size;
  }
}
