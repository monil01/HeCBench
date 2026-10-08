#ifndef BLACK_SCHOLES_IRIS_BLACK_SCHOLES_HPP
#define BLACK_SCHOLES_IRIS_BLACK_SCHOLES_HPP

#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948f
#endif

#ifndef M_SQRT_2
#define M_SQRT_2 0.7071067811865475244008443621048490392848359376887f
#endif

#define ERROR_FUNCT_tiny 0.000000000000000000001f
#define ERROR_FUNCT_one 1.00000000000000000000e+00f
#define ERROR_FUNCT_erx 8.45062911510467529297e-01f
#define ERROR_FUNCT_efx 1.28379167095512586316e-01f
#define ERROR_FUNCT_efx8 1.02703333676410069053e+00f
#define ERROR_FUNCT_pp0 1.28379167095512558561e-01f
#define ERROR_FUNCT_pp1 -3.25042107247001499370e-01f
#define ERROR_FUNCT_pp2 -2.84817495755985104766e-02f
#define ERROR_FUNCT_pp3 -5.77027029648944159157e-03f
#define ERROR_FUNCT_pp4 -2.37630166566501626084e-05f
#define ERROR_FUNCT_qq1 3.97917223959155352819e-01f
#define ERROR_FUNCT_qq2 6.50222499887672944485e-02f
#define ERROR_FUNCT_qq3 5.08130628187576562776e-03f
#define ERROR_FUNCT_qq4 1.32494738004321644526e-04f
#define ERROR_FUNCT_qq5 -3.96022827877536812320e-06f
#define ERROR_FUNCT_pa0 -2.36211856075265944077e-03f
#define ERROR_FUNCT_pa1 4.14856118683748331666e-01f
#define ERROR_FUNCT_pa2 -3.72207876035701323847e-01f
#define ERROR_FUNCT_pa3 3.18346619901161753674e-01f
#define ERROR_FUNCT_pa4 -1.10894694282396677476e-01f
#define ERROR_FUNCT_pa5 3.54783043256182359371e-02f
#define ERROR_FUNCT_pa6 -2.16637559486879084300e-03f
#define ERROR_FUNCT_qa1 1.06420880400844228286e-01f
#define ERROR_FUNCT_qa2 5.40397917702171048937e-01f
#define ERROR_FUNCT_qa3 7.18286544141962662868e-02f
#define ERROR_FUNCT_qa4 1.26171219808761642112e-01f
#define ERROR_FUNCT_qa5 1.36370839120290507362e-02f
#define ERROR_FUNCT_qa6 1.19844998467991074170e-02f
#define ERROR_FUNCT_ra0 -9.86494403484714822705e-03f
#define ERROR_FUNCT_ra1 -6.93858572707181764372e-01f
#define ERROR_FUNCT_ra2 -1.05586262253232909814e+01f
#define ERROR_FUNCT_ra3 -6.23753324503260060396e+01f
#define ERROR_FUNCT_ra4 -1.62396669462573470355e+02f
#define ERROR_FUNCT_ra5 -1.84605092906711035994e+02f
#define ERROR_FUNCT_ra6 -8.12874355063065934246e+01f
#define ERROR_FUNCT_ra7 -9.81432934416914548592e+00f
#define ERROR_FUNCT_sa1 1.96512716674392571292e+01f
#define ERROR_FUNCT_sa2 1.37657754143519042600e+02f
#define ERROR_FUNCT_sa3 4.34565877475229228821e+02f
#define ERROR_FUNCT_sa4 6.45387271733267880336e+02f
#define ERROR_FUNCT_sa5 4.29008140027567833386e+02f
#define ERROR_FUNCT_sa6 1.08635005541779435134e+02f
#define ERROR_FUNCT_sa7 6.57024977031928170135e+00f
#define ERROR_FUNCT_sa8 -6.04244152148580987438e-02f
#define ERROR_FUNCT_rb0 -9.86494292470009928597e-03f
#define ERROR_FUNCT_rb1 -7.99283237680523006574e-01f
#define ERROR_FUNCT_rb2 -1.77579549177547519889e+01f
#define ERROR_FUNCT_rb3 -1.60636384855821916062e+02f
#define ERROR_FUNCT_rb4 -6.37566443368389627722e+02f
#define ERROR_FUNCT_rb5 -1.02509513161107724954e+03f
#define ERROR_FUNCT_rb6 -4.83519191608651397019e+02f
#define ERROR_FUNCT_sb1 3.03380607434824582924e+01f
#define ERROR_FUNCT_sb2 3.25792512996573918826e+02f
#define ERROR_FUNCT_sb3 1.53672958608443695994e+03f
#define ERROR_FUNCT_sb4 3.19985821950859553908e+03f
#define ERROR_FUNCT_sb5 2.55305040643316442583e+03f
#define ERROR_FUNCT_sb6 4.74528541206955367215e+02f
#define ERROR_FUNCT_sb7 -2.24409524465858183362e+01f

#define THREAD_BLOCK_SIZE 256
#define CALL 0
#define PUT 1

struct payoffStruct {
  int type;
  float strike;
};

struct yieldTermStruct {
  float timeYearFraction;
  float forward;
  float compounding;
  float frequency;
  float intRate;
};

struct blackVolStruct {
  float timeYearFraction;
  float following;
  float volatility;
};

struct blackScholesMertStruct {
  float x0;
  yieldTermStruct dividendTS;
  yieldTermStruct riskFreeTS;
  blackVolStruct blackVolTS;
};

struct optionStruct {
  payoffStruct payoff;
  float yearFractionTime;
  blackScholesMertStruct pricingEngine;
};

struct blackCalcStruct {
  float strike;
  float forward;
  float stdDev;
  float discount;
  float variance;
  float d1;
  float d2;
  float alpha;
  float beta;
  float DalphaDd1;
  float DbetaDd2;
  float n_d1;
  float cum_d1;
  float n_d2;
  float cum_d2;
  float x;
  float DxDs;
  float DxDstrike;
};

struct normalDistStruct {
  float average;
  float sigma;
  float denominator;
  float derNormalizationFactor;
  float normalizationFactor;
};

struct optionInputStruct {
  int type;
  float strike;
  float spot;
  float q;
  float r;
  float t;
  float vol;
  float value;
  float tol;
};

#endif
