#include <iris/iris.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define _QUEENS_BLOCK_SIZE_ 128
#define _EMPTY_ -1

typedef struct queen_root {
  unsigned int control;
  int8_t board[12];
} QueenRoot;

inline void prefixesHandleSol(QueenRoot* root_prefixes, unsigned int flag,
                              const char* board, int initialDepth,
                              int num_sol) {
  root_prefixes[num_sol].control = flag;
  for (int i = 0; i < initialDepth; ++i) {
    root_prefixes[num_sol].board[i] = board[i];
  }
}

inline bool MCstillLegal(const char* board, const int r) {
  for (int i = 0; i < r; ++i)
    if (board[i] == board[r]) return false;
  int ld = board[r];
  int rd = board[r];
  for (int i = r - 1; i >= 0; --i) {
    --ld;
    ++rd;
    if (board[i] == ld || board[i] == rd) return false;
  }
  return true;
}

unsigned long long BP_queens_prefixes(int size, int initialDepth,
                                      unsigned long long* tree_size,
                                      QueenRoot* root_prefixes) {
  unsigned int flag = 0;
  int bit_test = 0;
  char vertice[20];
  unsigned long long local_tree = 0ULL;
  unsigned long long num_sol = 0;

  for (int i = 0; i < size; ++i) vertice[i] = -1;

  int nivel = 0;
  do {
    vertice[nivel]++;
    bit_test = 0;
    bit_test |= (1 << vertice[nivel]);

    if (vertice[nivel] == size) {
      vertice[nivel] = _EMPTY_;
    } else if (MCstillLegal(vertice, nivel) && !(flag & bit_test)) {
      flag |= (1ULL << vertice[nivel]);
      nivel++;
      ++local_tree;
      if (nivel == initialDepth) {
        prefixesHandleSol(root_prefixes, flag, vertice, initialDepth, num_sol);
        num_sol++;
      } else {
        continue;
      }
    } else {
      continue;
    }

    nivel--;
    flag &= ~(1ULL << vertice[nivel]);
  } while (nivel >= 0);

  *tree_size = local_tree;
  return num_sol;
}

void nqueens(short size, int initial_depth, unsigned int n_explorers,
             QueenRoot* root_prefixes_h,
             unsigned long long* vector_of_tree_size_h,
             unsigned long long* sols_h, const int repeat) {
  iris::DMem root_prefixes_d(root_prefixes_h, n_explorers * sizeof(QueenRoot));
  iris::DMem vector_of_tree_size_d(vector_of_tree_size_h,
                                   n_explorers * sizeof(unsigned long long));
  iris::DMem sols_d(sols_h, n_explorers * sizeof(unsigned long long));

  size_t gws[] = {static_cast<size_t>(
      std::ceil(static_cast<double>(n_explorers) / _QUEENS_BLOCK_SIZE_) *
      _QUEENS_BLOCK_SIZE_)};
  size_t lws[] = {_QUEENS_BLOCK_SIZE_};
  int N = size;
  int depth = initial_depth;

  std::printf("\n### Regular BP-DFS search. ###\n");
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    iris::Task task;
    void* params[] = {&N, &n_explorers, &depth, &root_prefixes_d,
                      &vector_of_tree_size_d, &sols_d};
    int pinfo[] = {sizeof(N), sizeof(n_explorers), sizeof(depth),
                   iris_r, iris_w, iris_w};
    task.kernel("BP_queens_root_dfs", 1, nullptr, gws, lws, 6, params, pinfo);
    task.flush_out(vector_of_tree_size_d);
    task.flush_out(sols_d);
    task.submit(iris_gpu, nullptr, 1);
  }
  auto end = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::printf("Average kernel execution time: %f (s)\n",
              (time * 1e-9f) / repeat);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::printf("Usage: %s <size> <initial depth> <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const short size = std::atoi(argv[1]);
  const int initialDepth = std::atoi(argv[2]);
  const int repeat = std::atoi(argv[3]);
  std::printf("\n### Initial depth: %d - Size: %d:", initialDepth, size);

  unsigned long long tree_size = 0ULL;
  unsigned long long qtd_sols_global = 0ULL;
  unsigned int nMaxPrefixos = 75580635;
  std::vector<QueenRoot> root_prefixes_h(nMaxPrefixos);

  unsigned long long n_explorers =
      BP_queens_prefixes(size, initialDepth, &tree_size, root_prefixes_h.data());
  root_prefixes_h.resize(n_explorers);
  std::vector<unsigned long long> vector_of_tree_size_h(n_explorers);
  std::vector<unsigned long long> solutions_h(n_explorers);

  nqueens(size, initialDepth, n_explorers, root_prefixes_h.data(),
          vector_of_tree_size_h.data(), solutions_h.data(), repeat);

  std::printf("\nTree size: %llu", tree_size);
  for (unsigned long long i = 0; i < n_explorers; ++i) {
    if (solutions_h[i] > 0) qtd_sols_global += solutions_h[i];
    if (vector_of_tree_size_h[i] > 0) tree_size += vector_of_tree_size_h[i];
  }

  std::printf("\nNumber of solutions found: %llu \nTree size: %llu\n",
              qtd_sols_global, tree_size);
  if (size == 15 && initialDepth == 7) {
    if (qtd_sols_global == 2279184 && tree_size == 171129071)
      std::printf("PASS\n");
    else
      std::printf("FAIL\n");
  }

  platform.finalize();
  return 0;
}
