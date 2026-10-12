#include <iris/iris.hpp>

#include <chrono>
#include <fstream>
#include <iostream>

#define BLOCK_SIZE 16

constexpr const int dim = 2;
using real = float;

template <int np, typename T>
using real_vector = T[np][np][dim];

template <int np, typename T>
using real_scalar = T[np][np];

template <int np, typename T>
struct element {
  real_scalar<np, T> metdet;
  T Dinv[np][np][2][2];
  real_scalar<np, T> rmetdet;
};

template <int np, typename T>
struct derivative {
  real_scalar<np, T> Dvv;
};

template <int np, typename T>
void readVelocity(T v[np][np][2], std::istream* input)
{
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < np; j++) {
      for (int k = 0; k < np; k++) {
        (*input) >> v[k][j][i];
      }
    }
  }
}

template <int np, typename T>
void readElement(element<np, T>& elem, std::istream* input)
{
  for (int i = 0; i < np; i++) {
    for (int j = 0; j < np; j++) {
      (*input) >> elem.metdet[i][j];
      elem.rmetdet[i][j] = 1 / elem.metdet[i][j];
    }
  }
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < np; k++) {
        for (int l = 0; l < np; l++) {
          (*input) >> elem.Dinv[l][k][i][j];
        }
      }
    }
  }
}

template <int np, typename T>
void readDerivative(derivative<np, T>& deriv, std::istream* input)
{
  for (int i = 0; i < np; i++) {
    for (int j = 0; j < np; j++) {
      (*input) >> deriv.Dvv[j][i];
    }
  }
}

template <int np, typename T>
void readDivergence(T divergence[np][np], std::istream* input)
{
  for (int i = 0; i < np; i++) {
    for (int j = 0; j < np; j++) {
      (*input) >> divergence[i][j];
    }
  }
}

template <int np, typename T>
void divergence_sphere_cpu(const real_vector<np, T> v,
                           const derivative<np, T>& deriv,
                           const element<np, T>& elem,
                           real_scalar<np, T> div)
{
  real_vector<np, T> gv;
  for (int j = 0; j < np; j++) {
    for (int i = 0; i < np; i++) {
      for (int k = 0; k < dim; k++) {
        gv[j][i][k] = elem.metdet[j][i] *
                      (elem.Dinv[j][i][k][0] * v[j][i][0] +
                       elem.Dinv[j][i][k][1] * v[j][i][1]);
      }
    }
  }

  real_scalar<np, T> vvtemp;
  for (int l = 0; l < np; l++) {
    for (int j = 0; j < np; j++) {
      T dudx00 = 0.0;
      T dvdy00 = 0.0;
      for (int i = 0; i < np; i++) {
        dudx00 += deriv.Dvv[l][i] * gv[j][i][0];
        dvdy00 += deriv.Dvv[l][i] * gv[i][j][1];
      }
      div[j][l] = dudx00;
      vvtemp[l][j] = dvdy00;
    }
  }

  constexpr T rrearth = 1.5683814303638645E-7;
  for (int i = 0; i < np; i++) {
    for (int j = 0; j < np; j++) {
      div[i][j] = (div[i][j] + vvtemp[i][j]) *
                  (elem.rmetdet[i][j] * rrearth);
    }
  }
}

template <int np, typename T>
void divergence_sphere_gpu(const real_vector<np, T> v,
                           const derivative<np, T>& deriv,
                           const element<np, T>& elem,
                           real_scalar<np, T> div)
{
  real_vector<np, T> gv;
  for (int j = 0; j < np; j++) {
    for (int i = 0; i < np; i++) {
      for (int k = 0; k < dim; k++) {
        gv[j][i][k] = elem.metdet[j][i] *
                      (elem.Dinv[j][i][k][0] * v[j][i][0] +
                       elem.Dinv[j][i][k][1] * v[j][i][1]);
      }
    }
  }

  T vvtemp[np][np] = {};
  iris::DMem d_gv(&gv[0][0][0], sizeof(T) * np * np * dim);
  iris::DMem d_Dvv(const_cast<T*>(&deriv.Dvv[0][0]), sizeof(T) * np * np);
  iris::DMem d_div(&div[0][0], sizeof(T) * np * np);
  iris::DMem d_vvtemp(&vvtemp[0][0], sizeof(T) * np * np);
  iris::DMem d_rmetdet(const_cast<T*>(&elem.rmetdet[0][0]), sizeof(T) * np * np);

  int np_arg = np;
  size_t gws[2] = {
      static_cast<size_t>(((np + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE),
      static_cast<size_t>(((np + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE)};
  size_t lws[2] = {BLOCK_SIZE, BLOCK_SIZE};
  void* params[6] = {&d_gv, &d_Dvv, &d_div, &d_vvtemp, &d_rmetdet, &np_arg};
  int pinfo[6] = {iris_r, iris_r, iris_rw, iris_rw, iris_r, sizeof(np_arg)};
  iris::Task task;
  task.kernel("div_kernel", 2, nullptr, gws, lws, 6, params, pinfo);
  task.flush_out(d_div);
  task.submit(iris_gpu, nullptr, 1);
  d_div.fetch(&div[0][0]);
}

template <int np, typename T>
void compareDivergences(const T v[np][np][dim],
                        const element<np, T>& elem,
                        const derivative<np, T>& deriv,
                        const T divergence_e[np][np],
                        const int numtests)
{
  std::cout << "Divergence on the CPU\n";
  T divergence_c[np][np];
  for (int i = 0; i < numtests; i++) {
    divergence_sphere_cpu<np, T>(v, deriv, elem, divergence_c);
  }

  auto start_c = std::chrono::steady_clock::now();
  for (int i = 0; i < numtests; i++) {
    divergence_sphere_cpu<np, T>(v, deriv, elem, divergence_c);
  }
  auto stop_c = std::chrono::steady_clock::now();
  auto time_c = std::chrono::duration_cast<std::chrono::microseconds>(stop_c - start_c).count();

  std::cout << "Divergence on the GPU\n";
  T divergence_f[np][np];
  for (int i = 0; i < numtests; i++) {
    divergence_sphere_gpu(v, deriv, elem, divergence_f);
  }

  auto start_f = std::chrono::steady_clock::now();
  for (int i = 0; i < numtests; i++) {
    divergence_sphere_gpu(v, deriv, elem, divergence_f);
  }
  auto stop_f = std::chrono::steady_clock::now();
  auto time_f = std::chrono::duration_cast<std::chrono::microseconds>(stop_f - start_f).count();

  std::cout << "Divergence Errors\n";
  std::cout << "CPU             GPU\n";
  for (int i = 0; i < np; i++) {
    for (int j = 0; j < np; j++) {
      std::cout << divergence_c[i][j] - divergence_e[i][j] << "    "
                << divergence_f[i][j] - divergence_e[i][j] << "\n";
    }
    std::cout << "\n";
  }

  std::cout << "Total CPU Time: " << time_c * 1e-3f << " (ms)" << std::endl;
  std::cout << "Total GPU Time: " << time_f * 1e-3f << " (ms)" << std::endl;
}

int main(int argc, char** argv)
{
  iris::Platform platform;
  platform.init(&argc, &argv, 1);

  constexpr const int NP = 4;
  real v[NP][NP][dim];
  element<NP, real> elem;
  derivative<NP, real> deriv;
  real divergence_e[NP][NP];
  {
    std::istream* input;
    if (argc > 1) {
      input = new std::ifstream(argv[1]);
    } else {
      input = &std::cin;
    }
    readVelocity(v, input);
    readElement(elem, input);
    readDerivative(deriv, input);
    readDivergence(divergence_e, input);
    if (argc > 1) {
      delete input;
    }
  }

  constexpr const int defNumTests = 100000;
  const int numtests = (argc > 2) ? std::stoi(argv[2]) : defNumTests;
  compareDivergences(v, elem, deriv, divergence_e, numtests);

  platform.finalize();
  return 0;
}
