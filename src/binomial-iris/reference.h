#pragma once

#include "binomialOptions.h"

#include <cmath>

static real CND(real d)
{
  const real A1 = 0.31938153;
  const real A2 = -0.356563782;
  const real A3 = 1.781477937;
  const real A4 = -1.821255978;
  const real A5 = 1.330274429;
  const real RSQRT2PI = 0.39894228040143267793994605993438;

  real K = 1.0 / (1.0 + 0.2316419 * std::fabs(d));
  real cnd = RSQRT2PI * std::exp(-0.5 * d * d) *
      (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

  if (d > 0)
    cnd = 1.0 - cnd;

  return cnd;
}

static void BlackScholesCall(real &callResult, TOptionData optionData)
{
  real S = optionData.S;
  real X = optionData.X;
  real T = optionData.T;
  real R = optionData.R;
  real V = optionData.V;

  real sqrtT = std::sqrt(T);
  real d1 = (std::log(S / X) + (R + (real)0.5 * V * V) * T) / (V * sqrtT);
  real d2 = d1 - V * sqrtT;
  real CNDD1 = CND(d1);
  real CNDD2 = CND(d2);

  real expRT = std::exp(-R * T);
  callResult = (real)(S * CNDD1 - X * expRT * CNDD2);
}

static real expiryCallValueCPU(real S, real X, real vDt, int i)
{
  real d = S * std::exp(vDt * (real)(2 * i - NUM_STEPS)) - X;
  return (d > (real)0) ? d : (real)0;
}

static void binomialOptionsCPU(real &callResult, TOptionData optionData)
{
  static real Call[NUM_STEPS + 1];

  const real S = optionData.S;
  const real X = optionData.X;
  const real T = optionData.T;
  const real R = optionData.R;
  const real V = optionData.V;

  const real dt = T / (real)NUM_STEPS;
  const real vDt = V * std::sqrt(dt);
  const real rDt = R * dt;
  const real If = std::exp(rDt);
  const real Df = std::exp(-rDt);
  const real u = std::exp(vDt);
  const real d = std::exp(-vDt);
  const real pu = (If - d) / (u - d);
  const real pd = 1.0 - pu;
  const real puByDf = pu * Df;
  const real pdByDf = pd * Df;

  for (int i = 0; i <= NUM_STEPS; i++)
    Call[i] = expiryCallValueCPU(S, X, vDt, i);

  for (int i = NUM_STEPS; i > 0; i--)
    for (int j = 0; j <= i - 1; j++)
      Call[j] = puByDf * Call[j + 1] + pdByDf * Call[j];

  callResult = (real)Call[0];
}
