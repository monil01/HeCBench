#pragma once

#include <cmath>
#include <cstdlib>

const double alpha = 0.02;
const double theta_S = 0.45;
const double theta_R = 0.1;
const double n = 1.8;

static void reference(
    const double* Ksat,
    const double* psi,
    double* C,
    double* theta,
    double* K,
    const int size)
{
  double Se, _theta, _psi, lambda, m;

  for (int i = 0; i < size; i++) {
    lambda = n - 1.0;
    m = lambda / n;

    _psi = psi[i] * 100;
    if (_psi < 0) {
      _theta = (theta_S - theta_R) / std::pow(1.0 + std::pow((alpha * (-_psi)), n), m) + theta_R;
    } else {
      _theta = theta_S;
    }

    theta[i] = _theta;
    Se = (_theta - theta_R) / (theta_S - theta_R);
    K[i] = Ksat[i] * std::sqrt(Se) * (1.0 - std::pow(1.0 - std::pow(Se, 1.0 / m), m)) *
           (1.0 - std::pow(1.0 - std::pow(Se, 1.0 / m), m));

    if (_psi < 0) {
      C[i] = 100 * alpha * n * (1.0 / n - 1.0) * std::pow(alpha * std::abs(_psi), n - 1) *
             (theta_R - theta_S) * std::pow(std::pow(alpha * std::abs(_psi), n) + 1, 1.0 / n - 2.0);
    } else {
      C[i] = 0.0;
    }
  }
}
