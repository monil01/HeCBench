#include <iris/iris.hpp>

#include "lebesgue.h"

#include <cstdlib>

double lebesgue_function(int n, double x[], int nfun, double xfun[])
{
  double lmax = 0.0;
  double* linterp = static_cast<double*>(std::malloc(n * nfun * sizeof(double)));
  if (!linterp) return lmax;

  iris::DMem d_max(&lmax, sizeof(double));
  iris::DMem d_interp(linterp, n * nfun * sizeof(double));
  iris::DMem d_x(x, n * sizeof(double));
  iris::DMem d_xfun(xfun, nfun * sizeof(double));

  size_t lws[1] = {256};
  size_t gws[1] = {static_cast<size_t>(((nfun + 255) / 256) * 256)};

  iris::Task task;
  void* params[6] = {&d_max, &d_interp, &d_xfun, &d_x, &n, &nfun};
  int pinfo[6] = {iris_rw, iris_w, iris_r, iris_r, sizeof(n), sizeof(nfun)};
  task.kernel("lebesgue_kernel", 1, nullptr, gws, lws, 6, params, pinfo);
  task.flush_out(d_max);
  task.submit(iris_gpu, nullptr, 1);
  d_max.fetch(&lmax);

  std::free(linterp);
  return lmax;
}
