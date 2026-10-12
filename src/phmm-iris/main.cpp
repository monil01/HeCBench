#include <iris/iris.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

#include "constants_types.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  const int repeat = atoi(argv[1]);

  size_t forward_matrix_size = (x_dim + 1) * (y_dim + 1) * batch * (states - 1) * sizeof(double);
  size_t emissions_size = (x_dim + 1) * (y_dim + 1) * batch * (states - 1) * sizeof(double);
  size_t transitions_size = (x_dim + 1) * (states - 1) * states * batch * sizeof(double);
  size_t start_transitions_size = batch * (states - 1) * sizeof(double);
  size_t likelihood_size = 2 * 2 * (states - 1) * batch * sizeof(double);

  fArray* h_cur_forward = static_cast<fArray*>(malloc(forward_matrix_size));
  fArray* h_next_forward = static_cast<fArray*>(malloc(forward_matrix_size));
  fArray* h_emis = static_cast<fArray*>(malloc(emissions_size));
  tArray* h_trans = static_cast<tArray*>(malloc(transitions_size));
  lArray* h_like = static_cast<lArray*>(malloc(likelihood_size));
  sArray* h_start = static_cast<sArray*>(malloc(start_transitions_size));

  std::default_random_engine rng(123);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (int i = 0; i < x_dim + 1; i++) {
    for (int j = 0; j < y_dim + 1; j++) {
      for (int b = 0; b < batch; b++) {
        for (int s = 0; s < states - 1; s++) {
          h_cur_forward[i][j][b][s] = dist(rng);
          h_next_forward[i][j][b][s] = h_cur_forward[i][j][b][s];
          h_emis[i][j][b][s] = dist(rng);
        }
      }
    }
  }

  for (int i = 0; i < x_dim + 1; i++) {
    for (int b = 0; b < batch; b++) {
      for (int s = 0; s < states - 1; s++) {
        for (int t = 0; t < states; t++) {
          h_trans[i][b][s][t] = dist(rng);
        }
      }
    }
  }

  for (int i = 0; i < batch; i++) {
    for (int s = 0; s < states - 1; s++) {
      h_start[i][s] = dist(rng);
    }
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int b = 0; b < batch; b++) {
        for (int s = 0; s < states - 1; s++) {
          h_like[i][j][b][s] = dist(rng);
        }
      }
    }
  }

  iris::DMem d_forward_a(h_cur_forward, forward_matrix_size);
  iris::DMem d_forward_b(h_next_forward, forward_matrix_size);
  iris::DMem d_emis(h_emis, emissions_size);
  iris::DMem d_trans(h_trans, transitions_size);
  iris::DMem d_like(h_like, likelihood_size);
  iris::DMem d_start(h_start, start_transitions_size);

  iris::DMem* d_cur_forward = &d_forward_a;
  iris::DMem* d_next_forward = &d_forward_b;

  size_t gws[1] = {batch * (states - 1)};
  size_t lws[1] = {states - 1};

  auto t1 = std::chrono::high_resolution_clock::now();

  for (int count = 0; count < repeat; count++) {
    for (int i = 1; i < x_dim + 1; i++) {
      for (int j = 1; j < y_dim + 1; j++) {
        iris::Task task;
        void* params[8] = {&i, &j, d_cur_forward, &d_trans, &d_emis, &d_like, &d_start, d_next_forward};
        int pinfo[8] = {sizeof(i), sizeof(j), iris_r, iris_r, iris_r, iris_r, iris_r, iris_w};
        task.kernel("pair_HMM_forward", 1, nullptr, gws, lws, 8, params, pinfo);
        if (count == repeat - 1 && i == x_dim && j == y_dim)
          task.flush_out(*d_next_forward);
        task.submit(iris_gpu, nullptr, 1);
        iris::DMem* t = d_cur_forward;
        d_cur_forward = d_next_forward;
        d_next_forward = t;
      }
    }
  }

  auto t2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> milli = (t2 - t1);
  std::cout << "Total execution time " << milli.count() << " milliseconds\n";

  d_cur_forward->fetch(h_cur_forward);

  double checkSum = 0.0;
  for (int i = 0; i < x_dim + 1; i++) {
    for (int j = 0; j < y_dim + 1; j++) {
      for (int b = 0; b < batch; b++) {
        for (int s = 0; s < states - 1; s++) {
#ifdef DEBUG
          std::cout << h_cur_forward[i][j][b][s] << std::endl;
#endif
          checkSum += h_cur_forward[i][j][b][s];
        }
      }
    }
  }
  std::cout << "Checksum " << checkSum << std::endl;

  free(h_cur_forward);
  free(h_next_forward);
  free(h_emis);
  free(h_trans);
  free(h_like);
  free(h_start);

  platform.finalize();
  return 0;
}
