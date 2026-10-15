#pragma once
// Original kernel signatures, with DMem base+byte-offset arguments.
inline void launch_scaledata(float2 *d_array, size_t array_size, float factor) {
  auto slice_0=tgv_dmem::slice(d_array);
  void* p[]={slice_0.mem,&array_size,&factor,&slice_0.offset};
  int pi[]={iris_rw,sizeof(array_size),sizeof(factor),sizeof(slice_0.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("scaledata",1,nullptr,g,l,4,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_arrayadd(float2 *array_c, const float2 *array_a, const float2 *array_b, size_t array_size, float alpha, float beta) {
  auto slice_0=tgv_dmem::slice(array_c);
  auto slice_1=tgv_dmem::slice(array_a);
  auto slice_2=tgv_dmem::slice(array_b);
  void* p[]={slice_0.mem,slice_1.mem,slice_2.mem,&array_size,&alpha,&beta,&slice_0.offset,&slice_1.offset,&slice_2.offset};
  int pi[]={iris_rw,iris_r,iris_r,sizeof(array_size),sizeof(alpha),sizeof(beta),sizeof(slice_0.offset),sizeof(slice_1.offset),sizeof(slice_2.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("arrayadd_complex",1,nullptr,g,l,9,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_arrayadd(float *array_c, const float *array_a, const float *array_b, size_t array_size, float alpha, float beta) {
  auto slice_0=tgv_dmem::slice(array_c);
  auto slice_1=tgv_dmem::slice(array_a);
  auto slice_2=tgv_dmem::slice(array_b);
  void* p[]={slice_0.mem,slice_1.mem,slice_2.mem,&array_size,&alpha,&beta,&slice_0.offset,&slice_1.offset,&slice_2.offset};
  int pi[]={iris_rw,iris_r,iris_r,sizeof(array_size),sizeof(alpha),sizeof(beta),sizeof(slice_0.offset),sizeof(slice_1.offset),sizeof(slice_2.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("arrayadd_real",1,nullptr,g,l,9,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_arraydot(float2 *array_c, const float2 *array_a, const float2 *array_b, size_t array_size, float alpha = 1.f) {
  auto slice_0=tgv_dmem::slice(array_c);
  auto slice_1=tgv_dmem::slice(array_a);
  auto slice_2=tgv_dmem::slice(array_b);
  void* p[]={slice_0.mem,slice_1.mem,slice_2.mem,&array_size,&alpha,&slice_0.offset,&slice_1.offset,&slice_2.offset};
  int pi[]={iris_rw,iris_r,iris_r,sizeof(array_size),sizeof(alpha),sizeof(slice_0.offset),sizeof(slice_1.offset),sizeof(slice_2.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("arraydot",1,nullptr,g,l,8,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_arrayreal(float2 *d_out, const float2 *d_in, size_t array_size) {
  auto slice_0=tgv_dmem::slice(d_out);
  auto slice_1=tgv_dmem::slice(d_in);
  void* p[]={slice_0.mem,slice_1.mem,&array_size,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_r,sizeof(array_size),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("arrayreal",1,nullptr,g,l,5,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_update_r(float2 *__restrict__ d_r, float2 *__restrict__ d_tmp, const float2 *d_imgb, size_t N, float sigma) {
  auto slice_0=tgv_dmem::slice(d_r);
  auto slice_1=tgv_dmem::slice(d_tmp);
  auto slice_2=tgv_dmem::slice(d_imgb);
  void* p[]={slice_0.mem,slice_1.mem,slice_2.mem,&N,&sigma,&slice_0.offset,&slice_1.offset,&slice_2.offset};
  int pi[]={iris_rw,iris_rw,iris_r,sizeof(N),sizeof(sigma),sizeof(slice_0.offset),sizeof(slice_1.offset),sizeof(slice_2.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("update_r",1,nullptr,g,l,8,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_xx(float2 *d_out, const float2 *d_in, size_t N, size_t rows, size_t cols, size_t ndyn) {
  auto slice_0=tgv_dmem::slice(d_out);
  auto slice_1=tgv_dmem::slice(d_in);
  void* p[]={slice_0.mem,slice_1.mem,&N,&rows,&cols,&ndyn,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_r,sizeof(N),sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_xx",1,nullptr,g,l,8,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_xx_bound(float2 *d_array, size_t N, size_t rows, size_t cols, size_t ndyn) {
  auto slice_0=tgv_dmem::slice(d_array);
  void* p[]={slice_0.mem,&N,&rows,&cols,&ndyn,&slice_0.offset};
  int pi[]={iris_rw,sizeof(N),sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(slice_0.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_xx_bound",1,nullptr,g,l,6,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_yy(float2 *d_out, const float2 *d_in, size_t N, size_t rows, size_t cols, size_t ndyn) {
  auto slice_0=tgv_dmem::slice(d_out);
  auto slice_1=tgv_dmem::slice(d_in);
  void* p[]={slice_0.mem,slice_1.mem,&N,&rows,&cols,&ndyn,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_r,sizeof(N),sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_yy",1,nullptr,g,l,8,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_yy_bound(float2 *d_array, size_t N, size_t rows, size_t cols, size_t ndyn) {
  auto slice_0=tgv_dmem::slice(d_array);
  void* p[]={slice_0.mem,&N,&rows,&cols,&ndyn,&slice_0.offset};
  int pi[]={iris_rw,sizeof(N),sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(slice_0.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_yy_bound",1,nullptr,g,l,6,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_tt(float2 *d_out, const float2 *d_in, size_t N, size_t rows, size_t cols, size_t ndyn, float mu) {
  auto slice_0=tgv_dmem::slice(d_out);
  auto slice_1=tgv_dmem::slice(d_in);
  void* p[]={slice_0.mem,slice_1.mem,&N,&rows,&cols,&ndyn,&mu,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_r,sizeof(N),sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(mu),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_tt",1,nullptr,g,l,9,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_tt_bound(float2 *d_array, size_t N, size_t rows, size_t cols, size_t ndyn) {
  auto slice_0=tgv_dmem::slice(d_array);
  void* p[]={slice_0.mem,&N,&rows,&cols,&ndyn,&slice_0.offset};
  int pi[]={iris_rw,sizeof(N),sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(slice_0.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_tt_bound",1,nullptr,g,l,6,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_proj_p(float2 *d_p, const float2 *d_tmp, const float2 *d_wbar, size_t N, float sigma, float alpha) {
  auto slice_0=tgv_dmem::slice(d_p);
  auto slice_1=tgv_dmem::slice(d_tmp);
  auto slice_2=tgv_dmem::slice(d_wbar);
  void* p[]={slice_0.mem,slice_1.mem,slice_2.mem,&N,&sigma,&alpha,&slice_0.offset,&slice_1.offset,&slice_2.offset};
  int pi[]={iris_rw,iris_r,iris_r,sizeof(N),sizeof(sigma),sizeof(alpha),sizeof(slice_0.offset),sizeof(slice_1.offset),sizeof(slice_2.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("proj_p",1,nullptr,g,l,9,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_xx_adj(float2 *d_out, const float2 *d_in, size_t N, size_t rows, size_t cols, size_t ndyn) {
  auto slice_0=tgv_dmem::slice(d_out);
  auto slice_1=tgv_dmem::slice(d_in);
  void* p[]={slice_0.mem,slice_1.mem,&N,&rows,&cols,&ndyn,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_r,sizeof(N),sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_xx_adj",1,nullptr,g,l,8,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_xx_adj_zero(float2 *d_out, const float2 *d_in, size_t rows, size_t cols, size_t ndyn) {
  auto slice_0=tgv_dmem::slice(d_out);
  auto slice_1=tgv_dmem::slice(d_in);
  void* p[]={slice_0.mem,slice_1.mem,&rows,&cols,&ndyn,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_r,sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_xx_adj_zero",1,nullptr,g,l,7,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_xx_adj_bound(float2 *d_out, const float2 *d_in, size_t rows, size_t cols, size_t ndyn) {
  auto slice_0=tgv_dmem::slice(d_out);
  auto slice_1=tgv_dmem::slice(d_in);
  void* p[]={slice_0.mem,slice_1.mem,&rows,&cols,&ndyn,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_r,sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_xx_adj_bound",1,nullptr,g,l,7,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_yy_adj(float2 *d_out, const float2 *d_in, size_t N, size_t rows, size_t cols, size_t ndyn) {
  auto slice_0=tgv_dmem::slice(d_out);
  auto slice_1=tgv_dmem::slice(d_in);
  void* p[]={slice_0.mem,slice_1.mem,&N,&rows,&cols,&ndyn,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_r,sizeof(N),sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_yy_adj",1,nullptr,g,l,8,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_yy_adj_zero(float2 *d_out, const float2 *d_in, size_t rows, size_t cols, size_t ndyn) {
  auto slice_0=tgv_dmem::slice(d_out);
  auto slice_1=tgv_dmem::slice(d_in);
  void* p[]={slice_0.mem,slice_1.mem,&rows,&cols,&ndyn,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_r,sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_yy_adj_zero",1,nullptr,g,l,7,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_yy_adj_bound(float2 *d_out, const float2 *d_in, size_t rows, size_t cols, size_t ndyn) {
  auto slice_0=tgv_dmem::slice(d_out);
  auto slice_1=tgv_dmem::slice(d_in);
  void* p[]={slice_0.mem,slice_1.mem,&rows,&cols,&ndyn,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_r,sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_yy_adj_bound",1,nullptr,g,l,7,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_tt_adj(float2 *d_out, const float2 *d_in, size_t N, size_t rows, size_t cols, size_t ndyn, float mu) {
  auto slice_0=tgv_dmem::slice(d_out);
  auto slice_1=tgv_dmem::slice(d_in);
  void* p[]={slice_0.mem,slice_1.mem,&N,&rows,&cols,&ndyn,&mu,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_r,sizeof(N),sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(mu),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_tt_adj",1,nullptr,g,l,9,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_tt_adj_zero(float2 *d_out, const float2 *d_in, size_t rows, size_t cols, size_t ndyn) {
  auto slice_0=tgv_dmem::slice(d_out);
  auto slice_1=tgv_dmem::slice(d_in);
  void* p[]={slice_0.mem,slice_1.mem,&rows,&cols,&ndyn,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_r,sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_tt_adj_zero",1,nullptr,g,l,7,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_grad_tt_adj_bound(float2 *d_out, float2 *d_in, size_t rows, size_t cols, size_t ndyn) {
  auto slice_0=tgv_dmem::slice(d_out);
  auto slice_1=tgv_dmem::slice(d_in);
  void* p[]={slice_0.mem,slice_1.mem,&rows,&cols,&ndyn,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_rw,sizeof(rows),sizeof(cols),sizeof(ndyn),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("grad_tt_adj_bound",1,nullptr,g,l,7,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_proj_q(float2 *__restrict__ d_q, float2 *__restrict__ d_tmp, size_t N, float sigma, float alpha) {
  auto slice_0=tgv_dmem::slice(d_q);
  auto slice_1=tgv_dmem::slice(d_tmp);
  void* p[]={slice_0.mem,slice_1.mem,&N,&sigma,&alpha,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_rw,sizeof(N),sizeof(sigma),sizeof(alpha),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("proj_q",1,nullptr,g,l,7,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_update_s(float2 *d_imgs, const float2 *d_tmp, const float2 *d_imgz, size_t N, float tau) {
  auto slice_0=tgv_dmem::slice(d_imgs);
  auto slice_1=tgv_dmem::slice(d_tmp);
  auto slice_2=tgv_dmem::slice(d_imgz);
  void* p[]={slice_0.mem,slice_1.mem,slice_2.mem,&N,&tau,&slice_0.offset,&slice_1.offset,&slice_2.offset};
  int pi[]={iris_rw,iris_r,iris_r,sizeof(N),sizeof(tau),sizeof(slice_0.offset),sizeof(slice_1.offset),sizeof(slice_2.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("update_s",1,nullptr,g,l,8,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_update_w(float2 *d_w, const float2 *d_tmp, const float2 *d_p, size_t N, float tau) {
  auto slice_0=tgv_dmem::slice(d_w);
  auto slice_1=tgv_dmem::slice(d_tmp);
  auto slice_2=tgv_dmem::slice(d_p);
  void* p[]={slice_0.mem,slice_1.mem,slice_2.mem,&N,&tau,&slice_0.offset,&slice_1.offset,&slice_2.offset};
  int pi[]={iris_rw,iris_r,iris_r,sizeof(N),sizeof(tau),sizeof(slice_0.offset),sizeof(slice_1.offset),sizeof(slice_2.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("update_w",1,nullptr,g,l,8,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_shrink(float2 *__restrict__ d_array2, float *__restrict__ d_array, float beta, int array_size) {
  auto slice_0=tgv_dmem::slice(d_array2);
  auto slice_1=tgv_dmem::slice(d_array);
  void* p[]={slice_0.mem,slice_1.mem,&beta,&array_size,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_rw,sizeof(beta),sizeof(array_size),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("shrink",1,nullptr,g,l,6,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}

inline void launch_arrayabs(float *d_array, const float2 *d_array2, size_t array_size) {
  auto slice_0=tgv_dmem::slice(d_array);
  auto slice_1=tgv_dmem::slice(d_array2);
  void* p[]={slice_0.mem,slice_1.mem,&array_size,&slice_0.offset,&slice_1.offset};
  int pi[]={iris_rw,iris_r,sizeof(array_size),sizeof(slice_0.offset),sizeof(slice_1.offset)};
  size_t g[]={size_t(gridsize)*size_t(blocksize)},l[]={size_t(blocksize)};
  iris::Task task;
  tgv_dmem::check(task.kernel("arrayabs",1,nullptr,g,l,5,p,pi));
  tgv_dmem::check(task.submit(0,nullptr,1));
}
