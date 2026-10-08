#include <cfloat>
#include <cmath>
#include <hip/hip_runtime.h>

#include "black_scholes.hpp"

__device__ float interestRateCompoundFactor(float t, yieldTermStruct currYieldTermStruct)
{
  return expf(currYieldTermStruct.forward * t);
}

__device__ float interestRateDiscountFactor(float t, yieldTermStruct currYieldTermStruct)
{
  return 1.0f / interestRateCompoundFactor(t, currYieldTermStruct);
}

__device__ float getBlackVolBlackVar(blackVolStruct volTS)
{
  float vol = volTS.volatility;
  return vol * vol * volTS.timeYearFraction;
}

__device__ float errorFunct(normalDistStruct, float x)
{
  float R, S, P, Q, s, y, z, r, ax;
  ax = fabsf(x);

  if (ax < 0.84375f) {
    if (ax < 3.7252902984e-09f) {
      if (ax < FLT_MIN * 16.0f) return 0.125f * (8.0f * x + ERROR_FUNCT_efx8 * x);
      return x + ERROR_FUNCT_efx * x;
    }
    z = x * x;
    r = ERROR_FUNCT_pp0 + z * (ERROR_FUNCT_pp1 + z * (ERROR_FUNCT_pp2 + z * (ERROR_FUNCT_pp3 + z * ERROR_FUNCT_pp4)));
    s = ERROR_FUNCT_one + z * (ERROR_FUNCT_qq1 + z * (ERROR_FUNCT_qq2 + z * (ERROR_FUNCT_qq3 + z * (ERROR_FUNCT_qq4 + z * ERROR_FUNCT_qq5))));
    y = r / s;
    return x + x * y;
  }
  if (ax < 1.25f) {
    s = ax - ERROR_FUNCT_one;
    P = ERROR_FUNCT_pa0 + s * (ERROR_FUNCT_pa1 + s * (ERROR_FUNCT_pa2 + s * (ERROR_FUNCT_pa3 + s * (ERROR_FUNCT_pa4 + s * (ERROR_FUNCT_pa5 + s * ERROR_FUNCT_pa6)))));
    Q = ERROR_FUNCT_one + s * (ERROR_FUNCT_qa1 + s * (ERROR_FUNCT_qa2 + s * (ERROR_FUNCT_qa3 + s * (ERROR_FUNCT_qa4 + s * (ERROR_FUNCT_qa5 + s * ERROR_FUNCT_qa6)))));
    return x >= 0.0f ? ERROR_FUNCT_erx + P / Q : -ERROR_FUNCT_erx - P / Q;
  }
  if (ax >= 6.0f) return x >= 0.0f ? ERROR_FUNCT_one - ERROR_FUNCT_tiny : ERROR_FUNCT_tiny - ERROR_FUNCT_one;

  s = ERROR_FUNCT_one / (ax * ax);
  if (ax < 2.85714285714285f) {
    R = ERROR_FUNCT_ra0 + s * (ERROR_FUNCT_ra1 + s * (ERROR_FUNCT_ra2 + s * (ERROR_FUNCT_ra3 + s * (ERROR_FUNCT_ra4 + s * (ERROR_FUNCT_ra5 + s * (ERROR_FUNCT_ra6 + s * ERROR_FUNCT_ra7))))));
    S = ERROR_FUNCT_one + s * (ERROR_FUNCT_sa1 + s * (ERROR_FUNCT_sa2 + s * (ERROR_FUNCT_sa3 + s * (ERROR_FUNCT_sa4 + s * (ERROR_FUNCT_sa5 + s * (ERROR_FUNCT_sa6 + s * (ERROR_FUNCT_sa7 + s * ERROR_FUNCT_sa8)))))));
  } else {
    R = ERROR_FUNCT_rb0 + s * (ERROR_FUNCT_rb1 + s * (ERROR_FUNCT_rb2 + s * (ERROR_FUNCT_rb3 + s * (ERROR_FUNCT_rb4 + s * (ERROR_FUNCT_rb5 + s * ERROR_FUNCT_rb6)))));
    S = ERROR_FUNCT_one + s * (ERROR_FUNCT_sb1 + s * (ERROR_FUNCT_sb2 + s * (ERROR_FUNCT_sb3 + s * (ERROR_FUNCT_sb4 + s * (ERROR_FUNCT_sb5 + s * (ERROR_FUNCT_sb6 + s * ERROR_FUNCT_sb7))))));
  }

  r = expf(-ax * ax - 0.5625f + R / S);
  return x >= 0.0f ? ERROR_FUNCT_one - r / ax : r / ax - ERROR_FUNCT_one;
}

__device__ float cumNormDistOp(normalDistStruct normDist, float z)
{
  z = (z - normDist.average) / normDist.sigma;
  return 0.5f * (1.0f + errorFunct(normDist, z * M_SQRT_2));
}

__device__ float gaussianFunctNormDist(normalDistStruct normDist, float x)
{
  float deltax = x - normDist.average;
  float exponent = -(deltax * deltax) / normDist.denominator;
  return exponent <= -690.0f ? 0.0f : normDist.normalizationFactor * expf(exponent);
}

__device__ float cumNormDistDeriv(normalDistStruct normDist, float x)
{
  float xn = (x - normDist.average) / normDist.sigma;
  return gaussianFunctNormDist(normDist, xn) / normDist.sigma;
}

__device__ void initCumNormDist(normalDistStruct& currCumNormDist)
{
  currCumNormDist.average = 0.0f;
  currCumNormDist.sigma = 1.0f;
  currCumNormDist.normalizationFactor = M_SQRT_2 * M_1_SQRTPI / currCumNormDist.sigma;
  currCumNormDist.derNormalizationFactor = currCumNormDist.sigma * currCumNormDist.sigma;
  currCumNormDist.denominator = 2.0f * currCumNormDist.derNormalizationFactor;
}

__device__ void initBlackCalcVars(blackCalcStruct& blackCalculator, payoffStruct payoff)
{
  blackCalculator.d1 = logf(blackCalculator.forward / blackCalculator.strike) / blackCalculator.stdDev + 0.5f * blackCalculator.stdDev;
  blackCalculator.d2 = blackCalculator.d1 - blackCalculator.stdDev;

  normalDistStruct currCumNormDist;
  initCumNormDist(currCumNormDist);

  blackCalculator.cum_d1 = cumNormDistOp(currCumNormDist, blackCalculator.d1);
  blackCalculator.cum_d2 = cumNormDistOp(currCumNormDist, blackCalculator.d2);
  blackCalculator.n_d1 = cumNormDistDeriv(currCumNormDist, blackCalculator.d1);
  blackCalculator.n_d2 = cumNormDistDeriv(currCumNormDist, blackCalculator.d2);

  blackCalculator.x = payoff.strike;
  blackCalculator.DxDstrike = 1.0f;
  blackCalculator.DxDs = 0.0f;

  switch (payoff.type) {
    case CALL:
      blackCalculator.alpha = blackCalculator.cum_d1;
      blackCalculator.DalphaDd1 = blackCalculator.n_d1;
      blackCalculator.beta = -blackCalculator.cum_d2;
      blackCalculator.DbetaDd2 = -blackCalculator.n_d2;
      break;
    case PUT:
      blackCalculator.alpha = -1.0f + blackCalculator.cum_d1;
      blackCalculator.DalphaDd1 = blackCalculator.n_d1;
      blackCalculator.beta = 1.0f - blackCalculator.cum_d2;
      blackCalculator.DbetaDd2 = -blackCalculator.n_d2;
      break;
  }
}

__device__ void initBlackCalculator(blackCalcStruct& blackCalc, payoffStruct payoff, float forwardPrice, float stdDev, float riskFreeDiscount)
{
  blackCalc.strike = payoff.strike;
  blackCalc.forward = forwardPrice;
  blackCalc.stdDev = stdDev;
  blackCalc.discount = riskFreeDiscount;
  blackCalc.variance = stdDev * stdDev;
  initBlackCalcVars(blackCalc, payoff);
}

__device__ float getResultVal(blackCalcStruct blackCalculator)
{
  return blackCalculator.discount * (blackCalculator.forward * blackCalculator.alpha + blackCalculator.x * blackCalculator.beta);
}

extern "C" __global__ void getOutValOption(const optionInputStruct* options, float* outputVals, int numVals)
{
  int optionNum = blockIdx.x * blockDim.x + threadIdx.x;
  if (optionNum >= numVals) return;

  const optionInputStruct threadOption = options[optionNum];

  payoffStruct currPayoff;
  currPayoff.type = threadOption.type;
  currPayoff.strike = threadOption.strike;

  yieldTermStruct qTS;
  qTS.timeYearFraction = threadOption.t;
  qTS.forward = threadOption.q;

  yieldTermStruct rTS;
  rTS.timeYearFraction = threadOption.t;
  rTS.forward = threadOption.r;

  blackVolStruct volTS;
  volTS.timeYearFraction = threadOption.t;
  volTS.volatility = threadOption.vol;

  blackScholesMertStruct stochProcess;
  stochProcess.x0 = threadOption.spot;
  stochProcess.dividendTS = qTS;
  stochProcess.riskFreeTS = rTS;
  stochProcess.blackVolTS = volTS;

  optionStruct currOption;
  currOption.payoff = currPayoff;
  currOption.yearFractionTime = threadOption.t;
  currOption.pricingEngine = stochProcess;

  float variance = getBlackVolBlackVar(currOption.pricingEngine.blackVolTS);
  float dividendDiscount = interestRateDiscountFactor(currOption.yearFractionTime, currOption.pricingEngine.dividendTS);
  float riskFreeDiscount = interestRateDiscountFactor(currOption.yearFractionTime, currOption.pricingEngine.riskFreeTS);
  float forwardPrice = currOption.pricingEngine.x0 * dividendDiscount / riskFreeDiscount;

  blackCalcStruct blackCalc;
  initBlackCalculator(blackCalc, currOption.payoff, forwardPrice, sqrtf(variance), riskFreeDiscount);
  outputVals[optionNum] = getResultVal(blackCalc);
}
