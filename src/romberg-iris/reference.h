//
// https://en.wikipedia.org/wiki/Romberg%27s_method
//

#ifndef ROMBERG_IRIS_REFERENCE_H
#define ROMBERG_IRIS_REFERENCE_H

#include <cmath>
#include <cstdio>

#ifdef DEBUG
static void dump_row(size_t i, double* R)
{
  std::printf("R[%2zu] = ", i);
  for (size_t j = 0; j <= i; ++j) {
    std::printf("%f ", R[j]);
  }
  std::printf("\n");
}
#endif

/*
 f: function to integrate
 a: lower limit
 b: upper limit
 acc: desired accuracy
*/
static double reference(double (*f)(double), double a, double b, size_t max_steps, double acc)
{
  double R1[max_steps], R2[max_steps];
  double* Rp = &R1[0];
  double* Rc = &R2[0];
  double h = (b - a);
  Rp[0] = (f(a) + f(b)) * h * .5;

#ifdef DEBUG
  dump_row(0, Rp);
#endif

  for (size_t i = 1; i < max_steps; ++i) {
    h /= 2.;
    double c = 0;
    size_t ep = 1 << (i - 1);
    for (size_t j = 1; j <= ep; ++j) {
      c += f(a + (2 * j - 1) * h);
    }
    Rc[0] = h * c + .5 * Rp[0];

    for (size_t j = 1; j <= i; ++j) {
      double n_k = std::pow(4, j);
      Rc[j] = (n_k * Rc[j - 1] - Rp[j - 1]) / (n_k - 1);
    }

#ifdef DEBUG
    dump_row(i, Rc);
#endif

    if (i > 1 && std::fabs(Rp[i - 1] - Rc[i]) < acc) {
      return Rc[i - 1];
    }

    double* rt = Rp;
    Rp = Rc;
    Rc = rt;
  }
  return Rp[max_steps - 1];
}

#endif
