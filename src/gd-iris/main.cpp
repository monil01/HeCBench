#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "utils.h"
#include "reference.h"


#include "dmem.hpp"
int main(int argc, char *argv[]) {
  port::Session session(&argc, &argv);
  if (argc != 5) {
    printf("Usage: %s <path to file> <lambda> <alpha> <repeat>\n", argv[0]);
    return 1;
  }
  const std::string file_path = argv[1];
  const float lambda = atof(argv[2]);
  const float alpha = atof(argv[3]);
  const int iters = atof(argv[4]);

  //store the problem data in variable A and the data is going to be normalized
  Classification_Data_CRS A;
  get_CRSM_from_svm(A, file_path);

  const int m = A.m; // observations
  const int n = A.n; // features

  std::vector<float> x(n, 0.f);
  std::vector<float> grad (n);

  float *d_x;
  port::allocate((void**)&d_x, n * sizeof(float));
  port::copy(d_x, x.data(), n * sizeof(float), port::h2d);

  float *d_grad;
  port::allocate((void**)&d_grad, n * sizeof(float));

  float *d_total_obj_val;
  port::allocate((void**)&d_total_obj_val, sizeof(float));

  float *d_l2_norm;
  port::allocate((void**)&d_l2_norm, sizeof(float));

  int *d_correct;
  port::allocate((void**)&d_correct, sizeof(int));

  int *d_row_ptr;
  port::allocate((void**)&d_row_ptr, A.row_ptr.size() * sizeof(int));
  port::copy(d_row_ptr, A.row_ptr.data(),
             A.row_ptr.size() * sizeof(int), port::h2d);

  int *d_col_index;
  port::allocate((void**)&d_col_index, A.col_index.size() * sizeof(int));
  port::copy(d_col_index, A.col_index.data(),
             A.col_index.size() * sizeof(int), port::h2d);

  float *d_value;
  port::allocate((void**)&d_value, A.values.size() * sizeof(float));
  port::copy(d_value, A.values.data(),
             A.values.size() * sizeof(float), port::h2d);

  int *d_y_label;
  port::allocate((void**)&d_y_label, A.y_label.size() * sizeof(int));
  port::copy(d_y_label, A.y_label.data(),
             A.y_label.size() * sizeof(int), port::h2d);

  dim3 grid ((m+255)/256);
  dim3 block (256);

  dim3 grid2 ((n+255)/256);
  dim3 block2 (256);

  float obj_val;
  float train_error;

  iris_synchronize();
  auto start = std::chrono::steady_clock::now();

  float total_obj_val;
  float l2_norm;
  int correct;

  for (int k = 0; k < iters; k++) {

    // reset the training status
    port::zero(d_total_obj_val, 0, sizeof(float));
    port::zero(d_l2_norm, 0, sizeof(float));
    port::zero(d_correct, 0, sizeof(int));

    //reset gradient vector
    port::zero(d_grad, 0, n * sizeof(float));

    // compute the total objective, correct rate, and gradient
    port::launch("compute", grid, block,
        d_x,
        d_grad,
        d_row_ptr,
        d_col_index,
        d_value,
        d_y_label,
        d_total_obj_val,
        d_correct,
        m
      );

    // update x (gradient does not need to be updated)
    port::launch("update_norm", grid2, block2, d_x, d_grad, d_l2_norm, m, n, lambda, alpha);
  }
  iris_synchronize();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Training time takes %lf (s) for %d iterations\n\n", time * 1e-9, iters);

  port::copy(&total_obj_val, d_total_obj_val, sizeof(float), port::d2h);
  port::copy(&l2_norm, d_l2_norm, sizeof(float), port::d2h);
  port::copy(&correct, d_correct, sizeof(int), port::d2h);

  obj_val = total_obj_val / (float)m + 0.5f * lambda * l2_norm;
  train_error = 1.f-(correct/(float)m);

  printf("object value = %f train_error = %f\n", obj_val, train_error);

  reference(A, x, grad, m, n, iters, alpha, lambda, obj_val, train_error);

  port::release(d_row_ptr);
  port::release(d_col_index);
  port::release(d_value);
  port::release(d_y_label);
  port::release(d_x);
  port::release(d_grad);
  port::release(d_total_obj_val);
  port::release(d_l2_norm);
  port::release(d_correct);

  return 0;
}
