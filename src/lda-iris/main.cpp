#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include "iris_dmem.hpp"


int main(int argc, char* argv[]) {

  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  IRISSession session(&argc,&argv);
  // repeat kernel execution
  const int repeat = atoi(argv[1]);

  int i;
  srand(123);

  const int num_topics = 1000;
  const int num_words  = 10266;
  const int block_cnt  = 500;
  const int num_indptr = block_cnt; // max: num_words
  const int block_dim  = 256;
  const int num_iters  = 64;
 
  std::vector<float> alpha(num_topics);
  for (i = 0; i < num_topics; i++)  alpha[i] = (float) rand() / (float) RAND_MAX;

  std::vector<float> beta(num_topics * num_words);
  for (i = 0; i < num_topics * num_words; i++)  beta[i] = (float) rand() / (float) RAND_MAX;

  std::vector<float> grad_alpha(num_topics * block_cnt, 0.0f);
  std::vector<float> new_beta(num_topics * num_words, 0.0f);
  std::vector<int> h_locks(num_words, 0);
  std::vector<float> gamma (num_indptr * num_topics);

  std::vector<int> indptr (num_indptr+1, 0);
  indptr[num_indptr] = num_words-1;
  for (i = num_indptr; i >= 1; i--) {
    int t = indptr[i] - 1 - (rand() % (num_words/num_indptr));
    if (t < 0) break;
    indptr[i-1] = t;
  }
  const int num_cols = num_words;

  std::vector<int> cols (num_cols);
  std::vector<float> counts (num_cols);

  for (i = 0; i < num_cols; i++) {
    cols[i] = i;
    counts[i] = 0.5f; // arbitrary
  }

  float *d_alpha, *d_beta, *d_grad_alpha, *d_new_beta;
  float *d_counts, *d_gamma, *d_vali_losses, *d_train_losses;
  int *d_locks, *d_cols, *d_indptr;
  bool *d_vali;

  iris_alloc((void**)&d_alpha, sizeof(float) * num_topics);
  iris_copy(d_alpha, alpha.data(), sizeof(float) * num_topics, IRISHostToDevice);

  iris_alloc((void**)&d_beta, sizeof(float) * num_topics * num_words);
  iris_copy(d_beta, beta.data(), sizeof(float) * num_topics * num_words, IRISHostToDevice);

  iris_alloc((void**)&d_grad_alpha, sizeof(float) * num_topics * block_cnt);
  iris_copy(d_grad_alpha, grad_alpha.data(), sizeof(float) * block_cnt * num_topics, IRISHostToDevice);

  iris_alloc((void**)&d_new_beta, sizeof(float) * num_topics * num_words);
  iris_copy(d_new_beta, new_beta.data(), sizeof(float) * num_topics * num_words, IRISHostToDevice);

  iris_alloc((void**)&d_locks, sizeof(int) * num_words);
  iris_copy(d_locks, h_locks.data(), sizeof(int) * num_words, IRISHostToDevice);
  
  iris_alloc((void**)&d_cols, sizeof(int) * num_cols);
  iris_copy(d_cols, cols.data(), sizeof(int) * num_cols, IRISHostToDevice);

  iris_alloc((void**)&d_indptr, sizeof(int) * (num_indptr + 1));
  iris_copy(d_indptr, indptr.data(), sizeof(int) * (num_indptr + 1), IRISHostToDevice);

  iris_alloc((void**)&d_vali, sizeof(bool) * num_cols);

  iris_alloc((void**)&d_counts, sizeof(float) * num_cols);
  iris_copy(d_counts, counts.data(), sizeof(float) * num_cols, IRISHostToDevice);

  // gamma will be initialized in the kernel
  iris_alloc((void**)&d_gamma, sizeof(float) * num_indptr * num_topics);

  // reset losses
  iris_alloc((void**)&d_train_losses, sizeof(float) * block_cnt);
  iris_set(d_train_losses, 0, sizeof(float) * block_cnt);

  iris_alloc((void**)&d_vali_losses, sizeof(float) * block_cnt);
  iris_set(d_vali_losses, 0, sizeof(float) * block_cnt);

  // store device results
  std::vector<float> train_losses(block_cnt), vali_losses(block_cnt);

  // training
  iris_set(d_vali, 0, sizeof(bool) * num_cols); 
  bool init_gamma = false;

  iris_sync();
  auto start = std::chrono::steady_clock::now();

  for (i = 0; i < repeat; i++) {
    init_gamma = (i == 0) ? true : false;
    iris_launch("EstepKernel", block_cnt, block_dim, 
      d_cols,
      d_indptr,
      d_vali,
      d_counts,
      init_gamma, num_cols, num_indptr, num_topics, num_iters,
      d_alpha,
      d_beta,
      d_gamma,
      d_grad_alpha,
      d_new_beta,
      d_train_losses,
      d_vali_losses,
      d_locks);
  }

  iris_sync();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (training): %f (s)\n", (time * 1e-9f) / repeat);

  // validation
  iris_set(d_vali, 0xFFFFFFFF, sizeof(bool) * num_cols); 

  iris_sync();
  start = std::chrono::steady_clock::now();

  for (i = 0; i < repeat; i++) {
    iris_launch("EstepKernel", block_cnt, block_dim, 
      d_cols,
      d_indptr,
      d_vali,
      d_counts,
      init_gamma, num_cols, num_indptr, num_topics, num_iters,
      d_alpha,
      d_beta,
      d_gamma,
      d_grad_alpha,
      d_new_beta,
      d_train_losses,
      d_vali_losses,
      d_locks);
  }

  iris_sync();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (validation): %f (s)\n", (time * 1e-9f) / repeat);

  iris_copy(vali_losses.data(), d_vali_losses, sizeof(float) * block_cnt, IRISDeviceToHost);
  iris_copy(train_losses.data(), d_train_losses, sizeof(float) * block_cnt, IRISDeviceToHost);

  float total_train_loss = std::accumulate(train_losses.begin(), train_losses.end(), 0.0f);
  float total_vali_loss = std::accumulate(vali_losses.begin(), vali_losses.end(), 0.0f);
  printf("Total train and validate loss: %f %f\n", total_train_loss, total_vali_loss);

  iris_free(d_cols);
  iris_free(d_indptr);
  iris_free(d_vali);
  iris_free(d_counts);
  iris_free(d_alpha);
  iris_free(d_beta);
  iris_free(d_gamma);
  iris_free(d_grad_alpha);
  iris_free(d_new_beta);
  iris_free(d_train_losses);
  iris_free(d_vali_losses);
  iris_free(d_locks);

  return 0;
}
