#include <cmath>
#include <cfloat>
#include <cstdio>
#include <vector>
#include <fstream>
#include <type_traits>
#define RESTRICT
#define C(q) C[(q)-1]
#define Y(q) Y[(q)-1]
#define RF(q) RF[(q)-1]
#define RB(q) RB[(q)-1]
#define EG(q) EG[(q)-1]
#define RKLOW(q) RKLOW[(q)-1]
#define WDOT(q) WDOT[(q)-1]
#define RKF(q) RKF[(q)-1]
#define RKR(q) RKR[(q)-1]
#define A(b,c) A[(b)*11+(c)-1]
#define ROP2(a) (RKF(a)-RKR(a))
template<class T> T DIV(T x,T y){return x*(1.0f/y);}
template<class T> T EXP(T x){return std::exp(x);}
template<class T> T EXP10(T x){if constexpr(std::is_same_v<T,float>)return exp10f(x);else return exp10(x);}
template<class T> T LOG(T x){return std::log(x);}
template<class T> T LOG10(T x){return std::log10(x);}
template<class T> T MIN(T x,T y){return std::fmin(x,y);}
template<class T> T MAX(T x,T y){return std::fmax(x,y);}
template<class t1,class t2,class t3,class t4,class t5> t1 polyx(t1 x,t2 c0,t3 c1,t4 c2,t5 c3){return (((c3*x+c2)*x+c1)*x+c0)*x;}
template<class real> void cpu_gr_base(const real* P, const real* T, const real* Y, real* C, real TCONV,
        real PCONV){

    const real TEMP = T[0]*TCONV;
    const real PRES = P[0]*PCONV;
    const real SMALL = FLT_MIN;

    real SUM, ctmp;

    SUM = (real)0.0;

    C(1)  = ctmp = Y(1) * (real)4.96046521e-1;
    SUM  += ctmp;
    C(2)  = ctmp = Y(2) * (real)9.92093043e-1;
    SUM  += ctmp;
    C(3)  = ctmp = Y(3) * (real)6.25023433e-2;
    SUM  += ctmp;
    C(4)  = ctmp = Y(4) * (real)3.12511716e-2;
    SUM  += ctmp;
    C(5)  = ctmp = Y(5) * (real)5.87980383e-2;
    SUM  += ctmp;
    C(6)  = ctmp = Y(6) * (real)5.55082499e-2;
    SUM  += ctmp;
    C(7)  = ctmp = Y(7) * (real)3.02968146e-2;
    SUM  += ctmp;
    C(8)  = ctmp = Y(8) * (real)2.93990192e-2;
    SUM  += ctmp;
    C(9)  = ctmp = Y(9) * (real)6.65112065e-2;
    SUM  += ctmp;
    C(10) = ctmp = Y(10)* (real)6.23323639e-2;
    SUM  += ctmp;
    C(11) = ctmp = Y(11)* (real)3.57008335e-2;
    SUM  += ctmp;
    C(12) = ctmp = Y(12)* (real)2.27221341e-2;
    SUM  += ctmp;
    C(13) = ctmp = Y(13)* (real)3.33039255e-2;
    SUM  += ctmp;
    C(14) = ctmp = Y(14)* (real)3.84050525e-2;
    SUM  += ctmp;
    C(15) = ctmp = Y(15)* (real)3.56453112e-2;
    SUM  += ctmp;
    C(16) = ctmp = Y(16)* (real)3.32556033e-2;
    SUM  += ctmp;
    C(17) = ctmp = Y(17)* (real)2.4372606e-2;
    SUM  += ctmp;
    C(18) = ctmp = Y(18)* (real)2.37882046e-2;
    SUM  += ctmp;
    C(19) = ctmp = Y(19)* (real)2.26996304e-2;
    SUM  += ctmp;
    C(20) = ctmp = Y(20)* (real)2.43467162e-2;
    SUM  += ctmp;
    C(21) = ctmp = Y(21)* (real)2.37635408e-2;
    SUM  += ctmp;
    C(22) = ctmp = Y(22)* (real)3.56972032e-2;
    SUM  += ctmp;

    SUM = DIV (PRES, (SUM * (TEMP) * (real)8.314510e7));

    for (unsigned k=1; k<=22; k++) {
        C(k) = MAX(C(k), SMALL) * SUM;
    }
}
template<class real> void cpu_ratt_kernel(const real* RESTRICT T, real* RESTRICT RF, real TCONV){

    const real TEMP = T[0]*TCONV;

    const register real ALOGT = LOG(TEMP);
    const register real TI = (real)1.0e0/(TEMP);
    const real TI2 = TI*TI;
    real TMP;

    RF(1) = EXP((real)3.20498617e1 -(real)7.25286183e3*TI);
    RF(2) = EXP((real)1.08197783e1 +(real)2.67e0*ALOGT - (real)3.16523284e3*TI);
    RF(3) = EXP((real)1.9190789e1  +(real)1.51e0*ALOGT - (real)1.72603317e3*TI);
    RF(4) = EXP((real)1.0482906e1  +(real)2.4e0*ALOGT + (real)1.06178717e3*TI);
    RF(5) = (real)1.e18*TI;
    RF(6) = EXP((real)3.90385861e1 -(real)6.e-1*ALOGT);
    RF(7) = EXP((real)4.55408762e1 -(real)1.25e0*ALOGT);
    RF(8) = (real)5.5e20*TI2;
    RF(9) = (real)2.2e22*TI2;
    RF(10) = (real)5.e17*TI;
    RF(11) = (real)1.2e17*TI;
    RF(12) = EXP((real)4.24761511e1 - (real)8.6e-1*ALOGT);
    RF(13) = EXP((real)4.71503141e1 - (real)1.72e0*ALOGT);
    RF(14) = EXP((real)4.42511034e1 - (real)7.6e-1*ALOGT);
    RF(15) = EXP((real)4.47046282e1 - (real)1.24e0*ALOGT);
    RF(16) = EXP((real)3.19350862e1 - (real)3.7e-1*ALOGT);
    RF(17) = EXP((real)2.90097872e1 - (real)3.37658384e2*TI);
    RF(18) = EXP((real)3.04404238e1 - (real)4.12637667e2*TI);
    RF(19) = EXP((real)3.18908801e1 - (real)1.50965e2*TI);
    RF(20) = (real)2.e13;
    RF(21) = EXP((real)3.14683206e1 + (real)2.51608334e2*TI);
    RF(22) = EXP((real)2.55908003e1 + (real)8.20243168e2*TI);
    RF(23) = EXP((real)3.36712758e1 - (real)6.03860001e3*TI);
    RF(24) = EXP((real)1.6308716e1  + (real)2.e0*ALOGT - (real)2.61672667e3*TI);
    RF(25) = EXP((real)2.99336062e1 - (real)1.81158e3*TI);
    RF(26) = EXP((real)1.60803938e1 + (real)2.e0*ALOGT - (real)2.01286667e3*TI);
    RF(27) = EXP((real)2.81906369e1 - (real)1.61029334e2*TI);
    RF(28) = EXP((real)3.39940492e1 - (real)4.81075134e3*TI);
    RF(29) = EXP((real)3.40312786e1 - (real)1.50965e3*TI);
    RF(30) = EXP((real)1.76783433e1 + (real)1.228e0*ALOGT - (real)3.52251667e1*TI);
    RF(31) = EXP((real)1.75767107e1 + (real)1.5e0*ALOGT - (real)4.00560467e4*TI);
    RF(32) = EXP((real)2.85473118e1 - (real)2.40537567e4*TI);
    RF(33) = EXP((real)3.26416564e1 - (real)1.18759134e4*TI);
    RF(34) = (real)5.7e13;
    RF(35) = (real)3.e13;
    RF(36) = EXP((real)1.85223344e1 + (real)1.79e0*ALOGT - (real)8.40371835e2*TI);
    RF(37) = EXP((real)2.93732401e1 + (real)3.79928584e2*TI);
    RF(38) = (real)3.3e13;
    RF(39) = (real)5.e13;
    RF(40) = EXP((real)2.88547965e1 - (real)3.47219501e2*TI);
    RF(41) = EXP((real)2.77171988e1 + (real)4.8e-1*ALOGT + (real)1.30836334e2*TI);
    RF(42) = (real)7.34e13;
    RF(43) = (real)3.e13;
    RF(44) = (real)3.e13;
    RF(45) = (real)5.e13;
    RF(46) = EXP((real)3.9769885e1  - (real)1.e0*ALOGT - (real)8.55468335e3*TI);
    RF(47) = EXP((real)2.96591694e1 - (real)2.01286667e2*TI);
    RF(48) = EXP((real)3.77576522e1 - (real)8.e-1*ALOGT);
    RF(49) = EXP((real)1.31223634e1 + (real)2.e0*ALOGT - (real)3.63825651e3*TI);
    RF(50) = (real)8.e13;
    TMP = EXP((real)-7.54825001e2*TI);
    RF(51) = (real)1.056e13 * TMP;
    RF(52) = (real)2.64e12 * TMP;
    RF(53) = (real)2.e13;
    RF(54) = EXP((real)1.62403133e1 +(real)2.e0*ALOGT -(real)1.50965e3*TI);
    RF(55) = (real)2.e13;
    RF(56) = EXP((real)2.74203001e1 +(real)5.e-1*ALOGT -(real)2.26950717e3*TI);
    RF(57) = (real)4.e13;
    RF(58) = (real)3.2e13;
    RF(59) = EXP((real)3.03390713e1 -(real)3.01930001e2*TI);
    RF(60) = (real)3.e13;
    RF(61) = (real)1.5e13;
    RF(62) = (real)1.5e13;
    RF(63) = (real)3.e13;
    RF(64) = (real)7.e13;
    RF(65) = (real)2.8e13;
    RF(66) = (real)1.2e13;
    RF(67) = (real)3.e13;
    RF(68) = (real)9.e12;
    RF(69) = (real)7.e12;
    RF(70) = (real)1.4e13;
    RF(71) = EXP((real)2.7014835e1  + (real)4.54e-1*ALOGT - (real)1.30836334e3*TI);
    RF(72) = EXP((real)2.38587601e1 + (real)1.05e0*ALOGT - (real)1.64803459e3*TI);
    RF(73) = EXP((real)3.12945828e1 - (real)1.781387e3*TI);
    RF(74) = EXP((real)2.19558261e1 + (real)1.18e0*ALOGT + (real)2.2493785e2*TI);
    RF(75) = EXP((real)3.22361913e1 - (real)2.01286667e4*TI);
    TMP = EXP( (real)-4.02573334e3*TI);
    RF(76) = (real)1.e12 * TMP;
    RF(127) = (real)5.e13 * TMP;
    RF(129) = (real)1.e13 * TMP;
    RF(77) = EXP((real)3.21806786e1 + (real)2.59156584e2*TI);
    RF(78) = EXP((real)3.70803784e1 - (real)6.3e-1*ALOGT - (real)1.92731984e2*TI);
    RF(79) = (real)8.43e13;
    RF(80) = EXP((real)1.78408622e1 + (real)1.6e0*ALOGT - (real)2.72743434e3*TI);
    RF(81) = (real)2.501e13;
    RF(82) = EXP((real)3.10595094e1 - (real)1.449264e4*TI);
    RF(83) = EXP((real)2.43067848e1 - (real)4.49875701e3*TI);
    RF(84) = (real)1.e12;
    RF(85) = (real)1.34e13;
    RF(86) = EXP((real)1.01064284e1 + (real)2.47e0*ALOGT - (real)2.60666234e3*TI);
    RF(87) = (real)3.e13;
    RF(88) = (real)8.48e12;
    RF(89) = (real)1.8e13;
    RF(90) = EXP((real)8.10772006e0 + (real)2.81e0*ALOGT - (real)2.94884967e3*TI);
    RF(91) = (real)4.e13;
    TMP = EXP((real)2.86833501e2*TI);
    RF(92) = (real)1.2e13 * TMP;
    RF(107) = (real)1.6e13 * TMP;
    RF(93) = EXP((real)3.75927776e1 - (real)9.7e-1*ALOGT - (real)3.11994334e2*TI);
    RF(94) = EXP((real)2.9238457e1 + (real)1.e-1*ALOGT - (real)5.33409668e3*TI);
    RF(95) = (real)5.e13;
    RF(96) = (real)2.e13;
    RF(97) = (real)3.2e13;
    RF(98) = (real)1.6e13;
    RF(99) = (real)1.e13;
    RF(100) = (real)5.e12;
    RF(101) = EXP((real)-2.84796532e1 + (real)7.6e0*ALOGT + (real)1.77635484e3*TI);
    RF(102) = EXP((real)2.03077504e1 + (real)1.62e0*ALOGT - (real)5.45486868e3*TI);
    RF(103) = EXP((real)2.07430685e1 + (real)1.5e0*ALOGT - (real)4.32766334e3*TI);
    RF(104) = EXP((real)1.84206807e1 + (real)1.6e0*ALOGT - (real)1.570036e3*TI);
    RF(105) = (real)6.e13;
    RF(106) = EXP((real)1.47156719e1 + (real)2.e0*ALOGT - (real)4.16160184e3*TI);
    RF(108) = (real)1.e14;
    RF(109) = (real)1.e14;
    RF(110) = EXP((real)2.81010247e1 - (real)4.29747034e2*TI);
    RF(111) = (real)5.e13;
    RF(112) = (real)3.e13;
    RF(113) = (real)1.e13;
    RF(114) = EXP((real)3.43156328e1 - (real)5.2e-1*ALOGT - (real)2.55382459e4*TI);
    RF(115) = EXP((real)1.97713479e1 + (real)1.62e0*ALOGT - (real)1.86432818e4*TI);
    TMP = EXP((real)2.e0*ALOGT - (real)9.56111669e2*TI );
    RF(116) = (real)1.632e7 * TMP;
    RF(117) = (real)4.08e6 * TMP;
    RF(118) = EXP((real)-8.4310155e0 + (real)4.5e0*ALOGT + (real)5.03216668e2*TI);
    RF(119) = EXP((real)-7.6354939e0 + (real)4.e0*ALOGT + (real)1.00643334e3*TI);
    RF(120) = EXP((real)1.61180957e1 + (real)2.e0*ALOGT - (real)3.01930001e3*TI);
    RF(121) = EXP((real)1.27430637e2 - (real)1.182e1*ALOGT - (real)1.79799315e4*TI);
    RF(122) = (real)1.e14;
    RF(123) = (real)1.e14;
    RF(124) = (real)2.e13;
    RF(125) = (real)1.e13;
    RF(126) = EXP((real)3.34301138e1 -(real)6.e-2*ALOGT - (real)4.27734167e3*TI);
    RF(128) = EXP((real)2.11287309e1 +(real)1.43e0*ALOGT - (real)1.35365284e3*TI);
    RF(130) = EXP((real)2.81906369e1 -(real)6.79342501e2*TI);
    TMP = EXP((real)-1.00643334e3*TI);
    RF(131) = (real)7.5e12 * TMP;
    RF(152) = (real)1.e13 * TMP;
    RF(186) = (real)2.e13 * TMP;
    RF(132) = EXP((real)2.94360258e1 + (real)2.7e-1*ALOGT - (real)1.40900667e2*TI);
    RF(133) = (real)3.e13;
    RF(134) = (real)6.e13;
    RF(135) = (real)4.8e13;
    RF(136) = (real)4.8e13;
    RF(137) = (real)3.011e13;
    RF(138) = EXP((real)1.41081802e1 +(real)1.61e0*ALOGT +(real)1.9293327e2*TI);
    RF(139) = EXP((real)2.64270483e1 +(real)2.9e-1*ALOGT -(real)5.53538334e0*TI);
    RF(140) = EXP((real)3.83674178e1 -(real)1.39e0*ALOGT -(real)5.08248834e2*TI);
    RF(141) = (real)1.e13;
    RF(142) = EXP((real)2.32164713e1 + (real)2.99917134e2*TI);
    RF(143) = (real)9.033e13;
    RF(144) = (real)3.92e11;
    RF(145) = (real)2.5e13;
    RF(146) = EXP((real)5.56675073e1 - (real)2.83e0*ALOGT - (real)9.36888792e3*TI);
    RF(147) = EXP((real)9.64601125e1 - (real)9.147e0*ALOGT - (real)2.36008617e4*TI);
    RF(148) = (real)1.e14;
    RF(149) = (real)9.e13;
    TMP = EXP((real)-2.01286667e3*TI);
    RF(150) = (real)2.e13 * TMP;
    RF(151) = (real)2.e13 * TMP;
    RF(153) = (real)1.4e11;
    RF(154) = (real)1.8e10;
    RF(155) = EXP((real)2.97104627e1 + (real)4.4e-1*ALOGT - (real)4.46705436e4*TI);
    RF(156) = EXP((real)2.77079822e1 + (real)4.54e-1*ALOGT - (real)9.15854335e2*TI);
    RF(157) = EXP((real)1.77414365e1 + (real)1.93e0*ALOGT - (real)6.51665585e3*TI);
    RF(158) = EXP((real)1.65302053e1 + (real)1.91e0*ALOGT - (real)1.88203034e3*TI);
    TMP = EXP( (real)1.83e0*ALOGT - (real)1.10707667e2*TI );
    RF(159) = (real)1.92e7 * TMP;
    RF(160) = (real)3.84e5 * TMP;
    RF(161) = EXP((real)1.50964444e1 + (real)2.e0*ALOGT - (real)1.25804167e3*TI);
    RF(162) = EXP((real)3.13734413e1 - (real)3.05955734e4*TI);
    RF(163) = EXP((real)2.83241683e1 - (real)7.04503335e3*TI);
    RF(164) = EXP((real)1.61180957e1 + (real)2.e0*ALOGT - (real)4.02573334e3*TI);
    RF(165) = EXP((real)3.06267534e1 - (real)3.01930001e3*TI);
    RF(166) = (real)5.e13;
    RF(167) = (real)5.e13;
    RF(168) = EXP((real)1.23327053e1 + (real)2.e0*ALOGT - (real)4.62959334e3*TI);
    RF(169) = EXP((real)2.65223585e1 - (real)3.87476834e3*TI);
    RF(170) = EXP((real)4.07945264e1 - (real)9.9e-1*ALOGT - (real)7.95082335e2*TI);
    RF(171) = (real)2.e12;
    RF(172) = (real)1.604e13;
    RF(173) = (real)8.02e13;
    RF(174) = (real)2.e10;
    RF(175) = (real)3.e11;
    RF(176) = (real)3.e11;
    RF(177) = (real)2.4e13;
    RF(178) = EXP((real)2.28865889e1 - (real)4.90133034e2*TI);
    RF(179) = (real)1.2e14;
    RF(180) = EXP((real)1.85604427e1 + (real)1.9e0*ALOGT  - (real)3.78922151e3*TI);
    RF(181) = EXP((real)1.83130955e1 + (real)1.92e0*ALOGT - (real)2.86330284e3*TI);
    RF(182) = EXP((real)1.50796373e1 + (real)2.12e0*ALOGT - (real)4.37798501e2*TI);
    RF(183) = EXP((real)3.13199006e1 + (real)2.76769167e2*TI);
    RF(184) = EXP((real)1.56303353e1 + (real)1.74e0*ALOGT - (real)5.25861418e3*TI);
    RF(185) = (real)2.e14;
    RF(187) = (real)2.66e12;
    RF(188) = (real)6.6e12;
    RF(189) = (real)6.e13;
    RF(190) = EXP((real)3.02187852e1 -(real)1.64083859e3*TI);
    RF(191) = EXP((real)5.11268757e1 -(real)2.39e0*ALOGT - (real)5.62596234e3*TI);
    RF(192) = EXP((real)1.20435537e1 +(real)2.5e0*ALOGT - (real)1.2530095e3*TI);
    RF(193) = EXP((real)1.86030023e1 +(real)1.65e0*ALOGT - (real)1.6455185e2*TI);
    RF(194) = EXP((real)1.73708586e1 +(real)1.65e0*ALOGT + (real)4.89126601e2*TI);
    RF(195) = EXP((real)2.59162227e1 +(real)7.e-1*ALOGT - (real)2.95891401e3*TI);
    RF(196) = EXP((real)1.49469127e1 +(real)2.e0*ALOGT + (real)1.49958567e2*TI);
    RF(197) = EXP((real)9.16951838e0 +(real)2.6e0*ALOGT - (real)6.99974385e3*TI);
    RF(198) = EXP((real)7.8845736e-1 +(real)3.5e0*ALOGT - (real)2.85575459e3*TI);
    RF(199) = EXP((real)5.65703751e1 -(real)2.92e0*ALOGT - (real)6.29272443e3*TI);
    RF(200) = (real)1.8e12;
    RF(201) = (real)9.6e13;
    RF(202) = (real)2.4e13;
    RF(203) = (real)9.e10;
    RF(204) = (real)2.4e13;
    RF(205) = (real)1.1e13;
    RF(206) = EXP((real)7.50436995e1 -(real)5.22e0*ALOGT -(real)9.93701954e3*TI);
}
template<class real> void cpu_rdsmh_kernel(const real* RESTRICT T, real* RESTRICT EG, real TCONV){

    const real TEMP = T[0]*TCONV;
    const real TLOG = LOG((TEMP));
    const real TI = (real)1.0e0/(TEMP);

    const register real TN1 = TLOG - (real)1.0;

    if ((TEMP) > (real)1.0e3)
    {
        EG(1) = EXP((real)-3.20502331e+00 + (real)9.50158922e+02*TI
                    + (real)3.33727920e+00*TN1 + polyx (TEMP,
                    (real)- 2.47012365e-05, (real)8.32427963e-08,
                    (real)- 1.49638662e-11, (real)1.00127688e-15));
        EG(2) = EXP((real)-4.46682914e-01 - (real)2.54736599e+04*TI
                    + (real)2.50000001e+00*TN1 + polyx (TEMP,
                    (real)- 1.15421486e-11, (real)2.69269913e-15,
                    (real)- 3.94596029e-19, (real)2.49098679e-23));
        EG(3) = EXP((real)4.78433864e+00 - (real)2.92175791e+04*TI
                    + (real)2.56942078e+00*TN1 + polyx (TEMP,
                    (real)- 4.29870569e-05, (real)6.99140982e-09,
                    (real)- 8.34814992e-13, (real)6.14168455e-17));
        EG(4) = EXP((real)5.45323129e+00 + (real)1.08845772e+03*TI
                    + (real)3.28253784e+00*TN1 + polyx (TEMP,
                    (real) 7.41543770e-04, (real)- 1.26327778e-07,
                    (real) 1.74558796e-11, (real)- 1.08358897e-15));
        EG(5) = EXP((real)4.47669610e+00 - (real)3.85865700e+03*TI
                    + (real)3.09288767e+00*TN1 + polyx (TEMP,
                    (real) 2.74214858e-04, (real)2.10842047e-08,
                    (real)- 7.32884630e-12, (real)5.87061880e-16));
        EG(6) = EXP((real)4.96677010e+00 + (real)3.00042971e+04*TI
                    + (real)3.03399249e+00*TN1 + polyx (TEMP,
                    + (real)1.08845902e-03, (real)-2.73454197e-08,
                    (real)- 8.08683225e-12, (real)8.41004960e-16));
        EG(7) = EXP((real)3.78510215e+00 - (real)1.11856713e+02*TI
                    + (real)4.01721090e+00*TN1 + polyx (TEMP,
                    + (real)1.11991007e-03, (real)-1.05609692e-07,
                    + (real)9.52053083e-12, (real)-5.39542675e-16));
        EG(8) = EXP((real)2.91615662e+00 + (real)1.78617877e+04*TI
                    + (real)4.16500285e+00*TN1 + polyx (TEMP,
                    + (real)2.45415847e-03, (real)-3.16898708e-07,
                    + (real)3.09321655e-11, (real)-1.43954153e-15));
        EG(9) = EXP((real)5.48497999e+00 - (real)7.10124364e+04*TI
                    + (real)2.87846473e+00*TN1 + polyx (TEMP,
                    + (real)4.85456841e-04, (real)2.40742758e-08,
                    (real)- 1.08906541e-11, (real)8.80396915e-16));
        EG(10) = EXP((real)6.17119324e+00 - (real)4.62636040e+04*TI
                     + (real)2.87410113e+00*TN1 + polyx (TEMP,
                     + (real)1.82819646e-03, (real)-2.34824328e-07,
                     + (real)2.16816291e-11, (real)-9.38637835e-16));
        EG(11) = EXP((real)8.62650169e+00 - (real)5.09259997e+04*TI
                     + (real)2.29203842e+00*TN1 + polyx (TEMP,
                     + (real)2.32794319e-03, (real)-3.35319912e-07,
                     + (real)3.48255000e-11, (real)-1.69858183e-15));
        EG(12) = EXP((real)8.48007179e+00 - (real)1.67755843e+04*TI
                     + (real)2.28571772e+00*TN1 + polyx (TEMP,
                     + (real)3.61995018e-03, (real)-4.97857247e-07,
                     + (real)4.96403870e-11, (real)-2.33577197e-15));
        EG(13) = EXP((real)1.84373180e+01 + (real)9.46834459e+03*TI
                     + (real)7.48514950e-02*TN1 + polyx (TEMP,
                     + (real)6.69547335e-03, (real)-9.55476348e-07,
                     + (real)1.01910446e-10, (real)-5.09076150e-15));
        EG(14) = EXP((real)7.81868772e+00 + (real)1.41518724e+04*TI
                     + (real)2.71518561e+00*TN1 + polyx (TEMP,
                     + (real)1.03126372e-03, (real)-1.66470962e-07,
                     + (real)1.91710840e-11, (real)-1.01823858e-15));
        EG(15) = EXP((real)2.27163806e+00 + (real)4.87591660e+04*TI
                     + (real)3.85746029e+00*TN1 + polyx (TEMP,
                     + (real)2.20718513e-03, (real)-3.69135673e-07,
                     + (real)4.36241823e-11, (real)-2.36042082e-15));
        EG(16) = EXP((real)9.79834492e+00 - (real)4.01191815e+03*TI
                     + (real)2.77217438e+00*TN1 + polyx (TEMP,
                     + (real)2.47847763e-03, (real)-4.14076022e-07,
                     + (real)4.90968148e-11, (real)-2.66754356e-15));
        EG(17) = EXP((real)1.36563230e+01 + (real)1.39958323e+04*TI
                     + (real)1.76069008e+00*TN1 + polyx (TEMP,
                     + (real)4.60000041e-03, (real)-7.37098022e-07,
                     + (real)8.38676767e-11, (real)-4.41927820e-15));
        EG(18) = EXP((real)2.92957500e+00 - (real)1.27832520e+02*TI
                     + (real)3.77079900e+00*TN1 + polyx (TEMP,
                     + (real)3.93574850e-03, (real)-4.42730667e-07,
                     + (real)3.28702583e-11, (real)-1.05630800e-15));
        EG(19) = EXP((real)-1.23028121e+00 - (real)2.59359992e+04*TI
                     + (real)4.14756964e+00*TN1 + polyx (TEMP,
                     + (real)2.98083332e-03, (real)-3.95491420e-07,
                     + (real)3.89510143e-11, (real)-1.80617607e-15));
        EG(20) = EXP((real)6.40237010e-01 - (real)4.83166880e+04*TI
                     + (real)4.27803400e+00*TN1 + polyx (TEMP,
                     + (real)2.37814020e-03, (real)-2.71683483e-07,
                     + (real)2.12190050e-11, (real)-7.44318950e-16));
        EG(21) = EXP((real)7.78732378e+00 - (real)3.46128739e+04*TI
                     + (real)3.01672400e+00*TN1 + polyx (TEMP,
                     + (real)5.16511460e-03, (real)-7.80137248e-07,
                     + (real)8.48027400e-11, (real)-4.31303520e-15));
        EG(22) = EXP((real)1.03053693e+01 - (real)4.93988614e+03*TI
                     + (real)2.03611116e+00*TN1 + polyx (TEMP,
                     + (real)7.32270755e-03, (real)-1.11846319e-06,
                     + (real)1.22685769e-10, (real)-6.28530305e-15));
        EG(23) = EXP((real)1.34624343e+01 - (real)1.28575200e+04*TI
                     + (real)1.95465642e+00*TN1 + polyx (TEMP,
                     + (real)8.69863610e-03, (real)-1.33034445e-06,
                     + (real)1.46014741e-10, (real)-7.48207880e-15));
        EG(24) = EXP((real)1.51156107e+01 + (real)1.14263932e+04*TI
                     + (real)1.07188150e+00*TN1 + polyx (TEMP,
                     + (real)1.08426339e-02, (real)-1.67093445e-06,
                     + (real)1.84510001e-10, (real)-9.50014450e-15));
        EG(25) = EXP((real)-3.93025950e+00 - (real)1.93272150e+04*TI
                     + (real)5.62820580e+00*TN1 + polyx (TEMP,
                     + (real)2.04267005e-03, (real)-2.65575783e-07,
                     + (real)2.38550433e-11, (real)-9.70391600e-16));
        EG(26) = EXP((real)6.32247205e-01 + (real)7.55105311e+03*TI
                     + (real)4.51129732e+00*TN1 + polyx (TEMP,
                     + (real)4.50179872e-03, (real)-6.94899392e-07,
                     + (real)7.69454902e-11, (real)-3.97419100e-15));
        EG(27) = EXP((real)-5.03208790e+00 - (real)4.90321780e+02*TI
                     + (real)5.97566990e+00*TN1 + polyx (TEMP,
                     + (real)4.06529570e-03, (real)-4.57270750e-07,
                     + (real)3.39192008e-11, (real)-1.08800855e-15));
        EG(28) = EXP((real)-3.48079170e+00 + (real)2.25931220e+04*TI
                     + (real)5.40411080e+00*TN1 + polyx (TEMP,
                     + (real)5.86152950e-03, (real)-7.04385617e-07,
                     + (real)5.69770425e-11, (real)-2.04924315e-15));
        EG(29) = EXP((real)-1.12430500e+01 - (real)1.74824490e+04*TI
                     + (real)6.50078770e+00*TN1 + polyx (TEMP,
                     + (real)7.16236550e-03, (real)-9.46360533e-07,
                     + (real)9.23400083e-11, (real)-4.51819435e-15));
        EG(30) = EXP((real)-1.33133500e+01 + (real)9.23570300e+02*TI
                     + (real)6.73225700e+00*TN1 + polyx (TEMP,
                     + (real)7.45417000e-03, (real)-8.24983167e-07,
                     + (real)6.01001833e-11, (real)-1.88310200e-15));
        EG(31) = EXP((real)-1.55152970e+01 - (real)7.97622360e+03*TI
                     + (real)7.70974790e+00*TN1 + polyx (TEMP,
                     + (real)8.01574250e-03, (real)-8.78670633e-07,
                     + (real)6.32402933e-11, (real)-1.94313595e-15));
    }
    else
    {
        EG(1) = EXP((real)6.83010238e-01 + (real)9.17935173e+02*TI
                    + (real)2.34433112e+00*TN1 + polyx (TEMP,
                    + (real)3.99026037e-03, (real)-3.24635850e-06,
                    + (real)1.67976745e-09, (real)-3.68805881e-13));
        EG(2) = EXP((real)-4.46682853e-01 - (real)2.54736599e+04*TI
                    + (real)2.50000000e+00*TN1 + polyx (TEMP,
                    + (real)3.52666409e-13, (real)-3.32653273e-16,
                    + (real)1.91734693e-19, (real)-4.63866166e-23));
        EG(3) = EXP((real)2.05193346e+00 - (real)2.91222592e+04*TI
                    + (real)3.16826710e+00*TN1 + polyx (TEMP,
                    (real)- 1.63965942e-03, (real)1.10717733e-06,
                    (real)- 5.10672187e-10, (real)1.05632986e-13));
        EG(4) = EXP((real)3.65767573e+00 + (real)1.06394356e+03*TI
                    + (real)3.78245636e+00*TN1 + polyx (TEMP,
                    (real)- 1.49836708e-03, (real)1.64121700e-06,
                    (real)- 8.06774591e-10, (real)1.62186419e-13));
        EG(5) = EXP((real)-1.03925458e-01 - (real)3.61508056e+03*TI
                    + (real)3.99201543e+00*TN1 + polyx (TEMP,
                    (real)- 1.20065876e-03, (real)7.69656402e-07,
                    (real)- 3.23427778e-10, (real)6.82057350e-14));
        EG(6) = EXP((real)-8.49032208e-01 + (real)3.02937267e+04*TI
                    + (real)4.19864056e+00*TN1 + polyx (TEMP,
                    (real)- 1.01821705e-03, (real)1.08673369e-06,
                    (real)- 4.57330885e-10, (real)8.85989085e-14));
        EG(7) = EXP((real)3.71666245e+00 - (real)2.94808040e+02*TI
                    + (real)4.30179801e+00*TN1 + polyx (TEMP,
                    (real)- 2.37456025e-03, (real)3.52638152e-06,
                    (real)- 2.02303245e-09, (real)4.64612562e-13));
        EG(8) = EXP((real)3.43505074e+00 + (real)1.77025821e+04*TI
                    + (real)4.27611269e+00*TN1 + polyx (TEMP,
                    (real)- 2.71411208e-04, (real)2.78892835e-06,
                    (real)- 1.79809011e-09, (real)4.31227182e-13));
        EG(9) = EXP((real)2.08401108e+00 - (real)7.07972934e+04*TI
                    + (real)3.48981665e+00*TN1 + polyx (TEMP,
                    + (real)1.61917771e-04, (real)-2.81498442e-07,
                    + (real)2.63514439e-10, (real)-7.03045335e-14));
        EG(10) = EXP((real)1.56253185e+00 - (real)4.60040401e+04*TI
                     + (real)3.76267867e+00*TN1 + polyx (TEMP,
                     + (real)4.84436072e-04, (real)4.65816402e-07,
                     (real)- 3.20909294e-10, (real)8.43708595e-14));
        EG(11) = EXP((real)-7.69118967e-01 - (real)5.04968163e+04*TI
                     + (real)4.19860411e+00*TN1 + polyx (TEMP,
                     (real)- 1.18330710e-03, (real)1.37216037e-06,
                     (real)- 5.57346651e-10, (real)9.71573685e-14));
        EG(12) = EXP((real)1.60456433e+00 - (real)1.64449988e+04*TI
                     + (real)3.67359040e+00*TN1 + polyx (TEMP,
                     + (real)1.00547588e-03, (real)9.55036427e-07,
                     (real)- 5.72597854e-10, (real)1.27192867e-13));
        EG(13) = EXP((real)-4.64130376e+00 + (real)1.02466476e+04*TI
                     + (real)5.14987613e+00*TN1 + polyx (TEMP,
                     (real)- 6.83548940e-03, (real)8.19667665e-06,
                     (real)- 4.03952522e-09, (real)8.33469780e-13));
        EG(14) = EXP((real)3.50840928e+00 + (real)1.43440860e+04*TI
                     + (real)3.57953347e+00*TN1 + polyx (TEMP,
                     (real)- 3.05176840e-04, (real)1.69469055e-07,
                     + (real)7.55838237e-11, (real)-4.52212249e-14));
        EG(15) = EXP((real)9.90105222e+00 + (real)4.83719697e+04*TI
                     + (real)2.35677352e+00*TN1 + polyx (TEMP,
                     + (real)4.49229839e-03, (real)-1.18726045e-06,
                     + (real)2.04932518e-10, (real)-7.18497740e-15));
        EG(16) = EXP((real)3.39437243e+00 - (real)3.83956496e+03*TI
                     + (real)4.22118584e+00*TN1 + polyx (TEMP,
                     (real)- 1.62196266e-03, (real)2.29665743e-06,
                     (real)- 1.10953411e-09, (real)2.16884433e-13));
        EG(17) = EXP((real)6.02812900e-01 + (real)1.43089567e+04*TI
                     + (real)4.79372315e+00*TN1 + polyx (TEMP,
                     (real)- 4.95416685e-03, (real)6.22033347e-06,
                     (real)- 3.16071051e-09, (real)6.58863260e-13));
        EG(18) = EXP((real)1.31521770e+01 - (real)9.78601100e+02*TI
                     + (real)2.10620400e+00*TN1 + polyx (TEMP,
                     + (real)3.60829750e-03, (real)8.89745333e-07,
                     (real)- 6.14803000e-10, (real)1.03780500e-13));
        EG(19) = EXP((real)1.39397051e+01 - (real)2.64289807e+04*TI
                     + (real)8.08681094e-01*TN1 + polyx (TEMP,
                     + (real)1.16807815e-02, (real)-5.91953025e-06,
                     + (real)2.33460364e-09, (real)-4.25036487e-13));
        EG(20) = EXP((real)5.92039100e+00 - (real)4.86217940e+04*TI
                     + (real)3.28154830e+00*TN1 + polyx (TEMP,
                     + (real)3.48823955e-03, (real)-3.97587400e-07,
                     (real)- 1.00870267e-10, (real)4.90947725e-14));
        EG(21) = EXP((real)8.51054025e+00 - (real)3.48598468e+04*TI
                     + (real)3.21246645e+00*TN1 + polyx (TEMP,
                     + (real)7.57395810e-04, (real)4.32015687e-06,
                     (real)- 2.98048206e-09, (real)7.35754365e-13));
        EG(22) = EXP((real)4.09733096e+00 - (real)5.08977593e+03*TI
                     + (real)3.95920148e+00*TN1 + polyx (TEMP,
                     (real)- 3.78526124e-03, (real)9.51650487e-06,
                     (real)- 5.76323961e-09, (real)1.34942187e-12));
        EG(23) = EXP((real)4.70720924e+00 - (real)1.28416265e+04*TI
                     + (real)4.30646568e+00*TN1 + polyx (TEMP,
                     (real)- 2.09329446e-03, (real)8.28571345e-06,
                     (real)- 4.99272172e-09, (real)1.15254502e-12));
        EG(24) = EXP((real)2.66682316e+00 + (real)1.15222055e+04*TI
                     + (real)4.29142492e+00*TN1 + polyx (TEMP,
                     (real)- 2.75077135e-03, (real)9.99063813e-06,
                     (real)- 5.90388571e-09, (real)1.34342886e-12));
        EG(25) = EXP((real)1.24904170e+01 - (real)2.00594490e+04*TI
                     + (real)2.25172140e+00*TN1 + polyx (TEMP,
                     + (real)8.82751050e-03, (real)-3.95485017e-06,
                     + (real)1.43964658e-09, (real)-2.53324055e-13));
        EG(26) = EXP((real)1.22156480e+01 + (real)7.04291804e+03*TI
                     + (real)2.13583630e+00*TN1 + polyx (TEMP,
                     + (real)9.05943605e-03, (real)-2.89912457e-06,
                     + (real)7.78664640e-10, (real)-1.00728807e-13));
        EG(27) = EXP((real)9.57145350e+00 - (real)1.52147660e+03*TI
                     + (real)3.40906240e+00*TN1 + polyx (TEMP,
                     + (real)5.36928700e-03, (real)3.15248750e-07,
                     + (real)5.96548592e-10, (real)1.43369255e-13));
        EG(28) = EXP((real)4.10301590e+00 + (real)2.15728780e+04*TI
                     + (real)4.72945950e+00*TN1 + polyx (TEMP,
                     (real)- 1.59664290e-03, (real)7.92248683e-06,
                     (real)- 4.78821758e-09, (real)1.09655560e-12));
        EG(29) = EXP((real)1.71732140e+01 - (real)1.92456290e+04*TI
                     + (real)1.36318350e+00*TN1 + polyx (TEMP,
                     + (real)9.90691050e-03, (real)2.08284333e-06,
                     (real)- 2.77962958e-09, (real)7.92328550e-13));
        EG(30) = EXP((real)1.61453400e+01 - (real)1.07482600e+03*TI
                     + (real)1.49330700e+00*TN1 + polyx (TEMP,
                     (real) 1.04625900e-02, (real)7.47799000e-07,
                     (real)- 1.39076000e-09, (real)3.57907300e-13));
        EG(31) = EXP((real)2.11360340e+01 - (real)1.03123460e+04*TI
                     + (real)1.04911730e+00*TN1 + polyx (TEMP,
                     (real) 1.30044865e-02, (real)3.92375267e-07,
                     (real)- 1.63292767e-09, (real)4.68601035e-13));
    }
}
template<class real> void cpu_ratt2_kernel(const real* RESTRICT T, const real* RESTRICT RF, real* RESTRICT RB,
        const real* RESTRICT EG, real TCONV){

    real TEMP = T[0]*TCONV;
    

    const register real SMALL_INV = (real)1e37;
    const real RU = (real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));

    register real rtemp_inv;

    rtemp_inv = DIV ((EG(2)*EG(4)), (EG(3)*EG(5)));
    RB(1) = RF(1) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(3)), (EG(2)*EG(5)));
    RB(2) = RF(2) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(5)), (EG(2)*EG(6)));
    RB(3) = RF(3) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(5)), (EG(3)*EG(6)));
    RB(4) = RF(4) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(5) = RF(5) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(6) = RF(6) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(7) = RF(7) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(2)*PFAC), EG(1));
    RB(8) = RF(8) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(5)*PFAC), EG(6));
    RB(9) = RF(9) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(3)*PFAC), EG(5));
    RB(10) = RF(10) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(3)*PFAC), EG(4));
    RB(11) = RF(11) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(12) = RF(12) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(13) = RF(13) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(14) = RF(14) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(4)*PFAC), EG(7));
    RB(15) = RF(15) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(5)*PFAC), EG(8));
    RB(16) = RF(16) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(7)), (EG(3)*EG(6)));
    RB(17) = RF(17) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(7)), (EG(1)*EG(4)));
    RB(18) = RF(18) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(7)), (EG(5)*EG(5)));
    RB(19) = RF(19) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(7)), (EG(4)*EG(5)));
    RB(20) = RF(20) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(7)), (EG(4)*EG(6)));
    RB(21) = RF(21) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(7)), (EG(4)*EG(8)));
    RB(22) = RF(22) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(7)), (EG(4)*EG(8)));
    RB(23) = RF(23) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(8)), (EG(1)*EG(7)));
    RB(24) = RF(24) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(8)), (EG(5)*EG(6)));
    RB(25) = RF(25) * MIN(rtemp_inv, SMALL_INV);
}
template<class real> void cpu_ratt3_kernel(const real* RESTRICT T, const real* RESTRICT RF, real* RESTRICT RB,
        const real* RESTRICT EG, real TCONV){

    real TEMP = T[0]*TCONV;
    
    const register real SMALL_INV = (real)1e37;
    const real RU=(real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));

    register real rtemp_inv;

    rtemp_inv = DIV ((EG(3)*EG(8)), (EG(5)*EG(7)));
    RB(26) = RF(26) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(8)), (EG(6)*EG(7)));
    RB(27) = RF(27) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(8)), (EG(6)*EG(7)));
    RB(28) = RF(28) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(14)*PFAC), EG(15));
    RB(29) = RF(29) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(14)), (EG(2)*EG(15)));
    RB(30) = RF(30) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(14)*PFAC), EG(17));
    RB(31) = RF(31) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(14)), (EG(3)*EG(15)));
    RB(32) = RF(32) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(14)), (EG(5)*EG(15)));
    RB(33) = RF(33) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(9)), (EG(2)*EG(14)));
    RB(34) = RF(34) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(9)), (EG(2)*EG(16)));
    RB(35) = RF(35) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(9)), (EG(2)*EG(10)));
    RB(36) = RF(36) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(6)*EG(9)), (EG(2)*EG(17)));
    RB(37) = RF(37) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(9)), (EG(3)*EG(16)));
    RB(38) = RF(38) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(14)*PFAC), EG(25));
    RB(39) = RF(39) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(15)), (EG(14)*EG(16)));
    RB(40) = RF(40) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(16)*PFAC), EG(17));
    RB(41) = RF(41) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(16)), (EG(1)*EG(14)));
    RB(42) = RF(42) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(16)), (EG(5)*EG(14)));
    RB(43) = RF(43) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(16)), (EG(2)*EG(15)));
    RB(44) = RF(44) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(16)), (EG(6)*EG(14)));
    RB(45) = RF(45) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*PFAC), (EG(2)*EG(14)));
    RB(46) = RF(46) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(16)), (EG(7)*EG(14)));
    RB(47) = RF(47) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(10)*PFAC), EG(12));
    RB(48) = RF(48) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(10)), (EG(2)*EG(12)));
    RB(49) = RF(49) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(10)), (EG(2)*EG(16)));
    RB(50) = RF(50) * MIN(rtemp_inv, SMALL_INV);
}
template<class real> void cpu_ratt4_kernel(const real* RESTRICT T, const real* RESTRICT RF, real* RESTRICT RB,
        const real* RESTRICT EG, real TCONV){

    real TEMP = T[0]*TCONV;
    
    const register real SMALL_INV = (real)1e37;
    const real RU=(real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));
    register real rtemp_inv;

    rtemp_inv = DIV ((EG(4)*EG(10)), (EG(5)*EG(16)));
    RB(51) = RF(51) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(10)), (EG(2)*EG(2)*EG(15)*PFAC));
    RB(52) = RF(52) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(10)), (EG(2)*EG(17)));
    RB(53) = RF(53) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(10)), (EG(6)*EG(9)));
    RB(54) = RF(54) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(10)), (EG(5)*EG(17)));
    RB(55) = RF(55) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(14)*PFAC), EG(26));
    RB(56) = RF(56) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(10)), (EG(2)*EG(19)));
    RB(57) = RF(57) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(10)), (EG(1)*EG(19)));
    RB(58) = RF(58) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(59) = RF(59) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(11)), (EG(1)*EG(9)));
    RB(60) = RF(60) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(11)), (EG(1)*EG(14)));
    RB(61) = RF(61) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(11)), (EG(2)*EG(16)));
    RB(62) = RF(62) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(11)), (EG(2)*EG(17)));
    RB(63) = RF(63) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(1)*EG(11)), (EG(2)*EG(12)));
    RB(64) = RF(64) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(11)), (EG(2)*EG(5)*EG(14)*PFAC));
    RB(65) = RF(65) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(11)), (EG(6)*EG(14)));
    RB(66) = RF(66) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(67) = RF(67) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(68) = RF(68) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(11), EG(10));
    RB(69) = RF(69) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(15)), (EG(14)*EG(17)));
    RB(70) = RF(70) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(17)*PFAC), EG(18));
    RB(71) = RF(71) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(17)), (EG(1)*EG(16)));
    RB(72) = RF(72) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(17)), (EG(5)*EG(16)));
    RB(73) = RF(73) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(17)), (EG(6)*EG(16)));
    RB(74) = RF(74) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(17)), (EG(7)*EG(16)));
    RB(75) = RF(75) * MIN(rtemp_inv, SMALL_INV);
}
template<class real> void cpu_ratt5_kernel(const real* RESTRICT T, const real* RESTRICT RF, real* RESTRICT RB,
        const real* RESTRICT EG, real TCONV){

    const real TEMP = T[0]*TCONV;
    
    const register real SMALL_INV = (real)1e37;
    const real RU=(real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));

    register real rtemp_inv;

    rtemp_inv = DIV ((EG(7)*EG(17)), (EG(8)*EG(16)));
    RB(76) = RF(76) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(17)), (EG(2)*EG(26)));
    RB(77) = RF(77) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(12)*PFAC), EG(13));
    RB(78) = RF(78) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(12)), (EG(2)*EG(17)));
    RB(79) = RF(79) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(12)), (EG(6)*EG(10)));
    RB(80) = RF(80) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(12)), (EG(6)*EG(11)));
    RB(81) = RF(81) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(12)), (EG(3)*EG(18)));
    RB(82) = RF(82) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(12)), (EG(5)*EG(17)));
    RB(83) = RF(83) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(12)), (EG(4)*EG(13)));
    RB(84) = RF(84) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(12)), (EG(5)*EG(18)));
    RB(85) = RF(85) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(8)*EG(12)), (EG(7)*EG(13)));
    RB(86) = RF(86) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(12)), (EG(2)*EG(21)));
    RB(87) = RF(87) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(16)), (EG(13)*EG(14)));
    RB(88) = RF(88) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(16)*PFAC), EG(28));
    RB(89) = RF(89) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(17)), (EG(13)*EG(16)));
    RB(90) = RF(90) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(12)), (EG(2)*EG(22)));
    RB(91) = RF(91) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(12)), (EG(2)*EG(22)));
    RB(92) = RF(92) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(12)*PFAC), EG(24));
    RB(93) = RF(93) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(12)), (EG(2)*EG(23)));
    RB(94) = RF(94) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(25)), (EG(14)*EG(22)));
    RB(95) = RF(95) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(18)), (EG(1)*EG(17)));
    RB(96) = RF(96) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(18)), (EG(5)*EG(12)));
    RB(97) = RF(97) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(18)), (EG(6)*EG(11)));
    RB(98) = RF(98) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(18)), (EG(5)*EG(17)));
    RB(99) = RF(99) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(18)), (EG(6)*EG(17)));
    RB(100) = RF(100) * MIN(rtemp_inv, SMALL_INV);

}
template<class real> void cpu_ratt6_kernel(const real* RESTRICT T, const real* RESTRICT RF, real* RESTRICT RB,
        const real* RESTRICT EG, real TCONV){

    const real TEMP = T[0]*TCONV;
    
    const register real SMALL_INV = (real)1e37;
    const real RU = (real)8.31451e7;
    const real PATM =  (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));
    register real rtemp_inv;

    rtemp_inv = DIV ((EG(4)*EG(18)), (EG(7)*EG(17)));
    RB(101) = RF(101) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(13)), (EG(1)*EG(12)));
    RB(102) = RF(102) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(13)), (EG(5)*EG(12)));
    RB(103) = RF(103) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(13)), (EG(6)*EG(12)));
    RB(104) = RF(104) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(13)), (EG(2)*EG(22)));
    RB(105) = RF(105) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(13)), (EG(12)*EG(12)));
    RB(106) = RF(106) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(13)), (EG(12)*EG(12)));
    RB(107) = RF(107) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(25)), (EG(11)*EG(14)));
    RB(108) = RF(108) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(25)), (EG(2)*EG(14)*EG(14)*PFAC));
    RB(109) = RF(109) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(25)), (EG(5)*EG(14)*EG(14)*PFAC));
    RB(110) = RF(110) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(9)*EG(25)), (EG(14)*EG(19)));
    RB(111) = RF(111) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(25)), (EG(14)*EG(21)));
    RB(112) = RF(112) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(25)*EG(25)), (EG(14)*EG(14)*EG(19)*PFAC));
    RB(113) = RF(113) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(19), EG(20));
    RB(114) = RF(114) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(21), (EG(2)*EG(19)*PFAC));
    RB(115) = RF(115) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(19)), (EG(2)*EG(25)));
    RB(116) = RF(116) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(19)), (EG(10)*EG(14)));
    RB(117) = RF(117) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(19)), (EG(2)*EG(26)));
    RB(118) = RF(118) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(19)), (EG(12)*EG(14)));
    RB(119) = RF(119) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(19)), (EG(14)*EG(21)));
    RB(120) = RF(120) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(19)*PFAC), EG(29));
    RB(121) = RF(121) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(19), EG(20));
    RB(122) = RF(122) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(20)), (EG(10)*EG(14)));
    RB(123) = RF(123) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(20)), (EG(2)*EG(26)));
    RB(124) = RF(124) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(20)), (EG(10)*EG(15)));
    RB(125) = RF(125) * MIN(rtemp_inv, SMALL_INV);
}
template<class real> void cpu_ratt7_kernel(const real* RESTRICT T, const real* RESTRICT RF, real* RESTRICT RB,
        const real* RESTRICT EG, real TCONV){
    const real TEMP = T[0]*TCONV;
    
    const register real SMALL_INV = (real)1e37;
    const real RU= (real)8.31451e7;
    const real PATM =  (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));
    register real rtemp_inv;

    rtemp_inv = DIV ((EG(2)*EG(26)*PFAC), EG(27));
    RB(126) = RF(126) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(26)), (EG(1)*EG(25)));
    RB(127) = RF(127) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(26)), (EG(12)*EG(14)));
    RB(128) = RF(128) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(26)), (EG(5)*EG(25)));
    RB(129) = RF(129) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(26)), (EG(10)*EG(15)));
    RB(130) = RF(130) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(26)), (EG(6)*EG(25)));
    RB(131) = RF(131) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(21)*PFAC), EG(22));
    RB(132) = RF(132) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(21)), (EG(1)*EG(19)));
    RB(133) = RF(133) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(21)), (EG(1)*EG(20)));
    RB(134) = RF(134) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(21)), (EG(2)*EG(26)));
    RB(135) = RF(135) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(21)), (EG(12)*EG(14)));
    RB(136) = RF(136) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(21)), (EG(6)*EG(19)));
    RB(137) = RF(137) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(21)), (EG(7)*EG(19)));
    RB(138) = RF(138) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(21)), (EG(3)*EG(27)));
    RB(139) = RF(139) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(21)), (EG(16)*EG(17)));
    RB(140) = RF(140) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(21)), (EG(5)*EG(27)));
    RB(141) = RF(141) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(8)*EG(21)), (EG(7)*EG(22)));
    RB(142) = RF(142) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(21)), (EG(14)*EG(22)));
    RB(143) = RF(143) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(21)), (EG(13)*EG(19)));
    RB(144) = RF(144) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(21)*PFAC), EG(30));
    RB(145) = RF(145) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(21)), (EG(2)*EG(29)));
    RB(146) = RF(146) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(27), (EG(12)*EG(14)*PFAC));
    RB(147) = RF(147) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(27)*PFAC), EG(28));
    RB(148) = RF(148) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(27)), (EG(12)*EG(16)));
    RB(149) = RF(149) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(27)), (EG(1)*EG(26)));
    RB(150) = RF(150) * MIN(rtemp_inv, SMALL_INV);
}
template<class real> void cpu_ratt8_kernel(const real* RESTRICT T, const real* RESTRICT RF,
        real* RESTRICT RB, const real* RESTRICT EG, real TCONV){

    const real TEMP = T[0]*TCONV;
    
    const register real SMALL_INV = (real)1e37;
    const real RU = (real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));
    register real rtemp_inv;

    rtemp_inv = DIV ((EG(3)*EG(27)), (EG(5)*EG(26)));
    RB(151) = RF(151) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(27)), (EG(6)*EG(26)));
    RB(152) = RF(152) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(27)), (EG(7)*EG(26)));
    RB(153) = RF(153) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(27)), (EG(5)*EG(14)*EG(17)*PFAC));
    RB(154) = RF(154) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV (EG(22), (EG(1)*EG(20)*PFAC));
    RB(155) = RF(155) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(22)*PFAC), EG(23));
    RB(156) = RF(156) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(22)), (EG(1)*EG(21)));
    RB(157) = RF(157) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(22)), (EG(5)*EG(21)));
    RB(158) = RF(158) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(22)), (EG(12)*EG(16)));
    RB(159) = RF(159) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(22)), (EG(10)*EG(17)));
    RB(160) = RF(160) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(22)), (EG(6)*EG(21)));
    RB(161) = RF(161) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(22)), (EG(7)*EG(21)));
    RB(162) = RF(162) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(22)), (EG(5)*EG(28)));
    RB(163) = RF(163) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(22)), (EG(14)*EG(23)));
    RB(164) = RF(164) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(10)*EG(22)), (EG(2)*EG(29)));
    RB(165) = RF(165) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(22)), (EG(13)*EG(20)));
    RB(166) = RF(166) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(22)), (EG(2)*EG(29)));
    RB(167) = RF(167) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(22)), (EG(13)*EG(21)));
    RB(168) = RF(168) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(22)*PFAC), EG(31));
    RB(169) = RF(169) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv =  DIV ((EG(2)*EG(23)*PFAC), EG(24));
    RB(170) = RF(170) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(23)), (EG(1)*EG(22)));
    RB(171) = RF(171) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(23)), (EG(12)*EG(17)));
    RB(172) = RF(172) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(23)), (EG(2)*EG(28)));
    RB(173) = RF(173) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(23)), (EG(7)*EG(22)));
    RB(174) = RF(174) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(23)), (EG(4)*EG(24)));
    RB(175) = RF(175) * MIN(rtemp_inv, SMALL_INV);
}
template<class real> void cpu_ratt9_kernel(const real* RESTRICT T, const real* RESTRICT RF,
        real* RESTRICT RB, const real* RESTRICT EG, real TCONV){

    const real TEMP = T[0]*TCONV;
    
    const register real SMALL_INV = (real)1e37;
    const real RU = (real)8.31451e7;
    const real PATM = (real)1.01325e6;
    const real PFAC = DIV (PATM, (RU*(TEMP)));
    register real rtemp_inv;

    rtemp_inv = DIV ((EG(7)*EG(23)), (EG(8)*EG(22)));
    RB(176) = RF(176) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(23)), (EG(5)*EG(12)*EG(17)*PFAC));
    RB(177) = RF(177) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(8)*EG(23)), (EG(7)*EG(24)));
    RB(178) = RF(178) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(23)), (EG(14)*EG(24)));
    RB(179) = RF(179) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(24)),  (EG(1)*EG(23)));
    RB(180) = RF(180) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(24)), (EG(5)*EG(23)));
    RB(181) = RF(181) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(24)), (EG(6)*EG(23)));
    RB(182) = RF(182) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(11)*EG(24)), (EG(12)*EG(23)));
    RB(183) = RF(183) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(24)), (EG(13)*EG(23)));
    RB(184) = RF(184) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(29)*PFAC), EG(30));
    RB(185) = RF(185) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(29)), (EG(13)*EG(20)));
    RB(186) = RF(186) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(29)), (EG(4)*EG(30)));
    RB(187) = RF(187) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(29)), (EG(5)*EG(17)*EG(21)*PFAC));
    RB(188) = RF(188) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(16)*EG(29)), (EG(14)*EG(30)));
    RB(189) = RF(189) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(30)*PFAC), EG(31));
    RB(190) = RF(190) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(30)), (EG(12)*EG(22)));
    RB(191) = RF(191) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(30)), (EG(1)*EG(29)));
    RB(192) = RF(192) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(30)), (EG(2)*EG(12)*EG(26)*PFAC));
    RB(193) = RF(193) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(30)), (EG(16)*EG(23)));
    RB(194) = RF(194) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(30)), (EG(5)*EG(29)));
    RB(195) = RF(195) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(30)), (EG(6)*EG(29)));
    RB(196) = RF(196) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(30)), (EG(8)*EG(29)));
    RB(197) = RF(197) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(30)), (EG(13)*EG(29)));
    RB(198) = RF(198) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(31)), (EG(12)*EG(23)));
    RB(199) = RF(199) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(2)*EG(31)), (EG(1)*EG(30)));
    RB(200) = RF(200) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(3)*EG(31)), (EG(17)*EG(23)));
    RB(201) = RF(201) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(5)*EG(31)), (EG(6)*EG(30)));
    RB(202) = RF(202) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(4)*EG(31)), (EG(7)*EG(30)));
    RB(203) = RF(203) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(7)*EG(31)), (EG(5)*EG(17)*EG(23)*PFAC));
    RB(204) = RF(204) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(12)*EG(31)), (EG(13)*EG(30)));
    RB(205) = RF(205) * MIN(rtemp_inv, SMALL_INV);

    rtemp_inv = DIV ((EG(21)*EG(23)), (EG(12)*EG(29)));
    RB(206) = RF(206) * MIN(rtemp_inv, SMALL_INV);
}
template<class real> void cpu_ratt10_kernel(const real* RESTRICT T, real* RESTRICT RKLOW, real TCONV){

    const register real TEMP = T[0]*TCONV;
    const register real ALOGT = LOG(TEMP);

    RKLOW(1) = EXP((real)4.22794408e1  -(real)9.e-1*ALOGT   + DIV((real)8.55468335e2,TEMP));
    RKLOW(2) = EXP((real)6.37931383e1  -(real)3.42e0*ALOGT  - DIV((real)4.24463259e4,TEMP));
    RKLOW(3) = EXP((real)6.54619238e1  -(real)3.74e0*ALOGT  - DIV((real)9.74227469e2,TEMP));
    RKLOW(4) = EXP((real)5.55621468e1  -(real)2.57e0*ALOGT  - DIV((real)7.17083751e2,TEMP));
    RKLOW(5) = EXP((real)6.33329483e1  -(real)3.14e0*ALOGT  - DIV((real)6.18956501e2,TEMP));
    RKLOW(6) = EXP((real)7.69748493e1  -(real)5.11e0*ALOGT  - DIV((real)3.57032226e3,TEMP));
    RKLOW(7) = EXP((real)6.98660102e1  -(real)4.8e0*ALOGT   - DIV((real)2.79788467e3,TEMP));
    RKLOW(8) = EXP((real)7.68923562e1  -(real)4.76e0*ALOGT  - DIV((real)1.22784867e3,TEMP));
    RKLOW(9) = EXP((real)1.11312542e2  -(real)9.588e0*ALOGT - DIV((real)2.566405e3,TEMP));
    RKLOW(10) = EXP((real)1.15700234e2 -(real)9.67e0*ALOGT  - DIV((real)3.13000767e3,TEMP));
    RKLOW(11) = EXP((real)3.54348644e1 -(real)6.4e-1*ALOGT  - DIV((real)2.50098684e4,TEMP));
    RKLOW(12) = EXP((real)6.3111756e1  -(real)3.4e0*ALOGT   - DIV((real)1.80145126e4,TEMP));
    RKLOW(13) = EXP((real)9.57409899e1 -(real)7.64e0*ALOGT  - DIV((real)5.98827834e3,TEMP));
    RKLOW(14) = EXP((real)6.9414025e1  -(real)3.86e0*ALOGT  - DIV((real)1.67067934e3,TEMP));
    RKLOW(15) = EXP((real)1.35001549e2 -(real)1.194e1*ALOGT - DIV((real)4.9163262e3,TEMP));
    RKLOW(16) = EXP((real)9.14494773e1 -(real)7.297e0*ALOGT - DIV((real)2.36511834e3,TEMP));
    RKLOW(17) = EXP((real)1.17075165e2 -(real)9.31e0*ALOGT  - DIV((real)5.02512164e4,TEMP));
    RKLOW(18) = EXP((real)9.68908955e1 -(real)7.62e0*ALOGT  - DIV((real)3.50742017e3,TEMP));
    RKLOW(19) = EXP((real)9.50941235e1 -(real)7.08e0*ALOGT  - DIV((real)3.36400342e3,TEMP));
    RKLOW(20) = EXP((real)1.38440285e2 -(real)1.2e1*ALOGT   - DIV((real)3.00309643e3,TEMP));
    RKLOW(21) = EXP((real)8.93324137e1 -(real)6.66e0*ALOGT  - DIV((real)3.52251667e3,TEMP));
}
template<class real> void cpu_ratx_kernel(const real* RESTRICT T, const real* RESTRICT C, real* RESTRICT RF,
        real* RESTRICT RB, const real* RESTRICT RKLOW, real TCONV){
    const real TEMP = T[0]*TCONV;
    
    real CTOT = 0.0;
    register real PR, PCOR, PRLOG, FCENT, FCLOG, XN;
    register real CPRLOG, FLOG, FC;
    register real SQR;
    const real SMALL = FLT_MIN;

    for (unsigned int k=1; k<=22; k++) {
        CTOT += C(k);
    }

    real CTB_10 = CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11) + C(12)
                + (real)2.e0*C(16) + (real)2.e0*C(14) + (real)2.e0*C(15);
    real CTB_114= CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11) + C(12)
                + (real)2.e0*C(16) + (real)1.5e0*C(14) + (real)1.5e0*C(15) ;
    real CTB_16 = CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11) + C(12)
                + (real)2.e0*C(16) + (real)2.e0*C(14) + (real)2.e0*C(15) ;

    

    PR = RKLOW(1) * DIV(CTB_16, RF(16));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.654e-1*EXP(DIV(-TEMP,(real)9.4e1)) + (real)7.346e-1*EXP(DIV(-TEMP,(real)1.756e3))
    + EXP(DIV((real)-5.182e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(16) = RF(16) * PCOR;
    RB(16) = RB(16) * PCOR;

    PR = RKLOW(2) * DIV(CTB_10, RF(31));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)6.8e-2*EXP(DIV(-TEMP,(real)1.97e2)) + (real)9.32e-1*EXP(DIV(-TEMP,(real)1.54e3))
    + EXP(DIV((real)-1.03e4,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV (CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV (FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(31) = RF(31) * PCOR;
    RB(31) = RB(31) * PCOR;

    PR = RKLOW(3) * DIV(CTB_10, RF(39));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)4.243e-1*EXP(DIV(-TEMP,(real)2.37e2)) + (real)5.757e-1*EXP(DIV(-TEMP,(real)1.652e3))
    + EXP(DIV((real)-5.069e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(39) = RF(39) * PCOR;
    RB(39) = RB(39) * PCOR;

    PR = RKLOW(4) * DIV(CTB_10, RF(41));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.176e-1*EXP(DIV(-TEMP,(real)2.71e2)) + (real)7.824e-1*EXP(DIV(-TEMP,(real)2.755e3))
    + EXP(DIV((real)-6.57e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(41) = RF(41) * PCOR;
    RB(41) = RB(41) * PCOR;

    PR = RKLOW(5) * DIV(CTB_10, RF(48));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)3.2e-1*EXP(DIV(-TEMP,(real)7.8e1)) + (real)6.8e-1*EXP(DIV(-TEMP,(real)1.995e3))
    + EXP(DIV((real)-5.59e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(48) = RF(48) * PCOR;
    RB(48) = RB(48) * PCOR;

    PR = RKLOW(6) * DIV(CTB_10, RF(56));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)4.093e-1*EXP(DIV(-TEMP,(real)2.75e2)) + (real)5.907e-1*EXP(DIV(-TEMP,(real)1.226e3))
    + EXP(DIV((real)-5.185e3, TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(56) = RF(56) * PCOR;
    RB(56) = RB(56) * PCOR;

    PR = RKLOW(7) * DIV(CTB_10, RF(71));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.42e-1*EXP(DIV(-TEMP,(real)9.4e1)) + (real)7.58e-1*EXP(DIV(-TEMP,(real)1.555e3))
    + EXP(DIV((real)-4.2e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(71) = RF(71) * PCOR;
    RB(71) = RB(71) * PCOR;

    PR = RKLOW(8) * DIV(CTB_10, RF(78));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.17e-1*EXP(DIV(-TEMP,(real)7.4e1)) + (real)7.83e-1*EXP(DIV(-TEMP,(real)2.941e3))
    + EXP(DIV((real)-6.964e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(78) = RF(78) * PCOR;
    RB(78) = RB(78) * PCOR;

    PR = RKLOW(9) * DIV(CTB_10, RF(89));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)3.827e-1*EXP(DIV(-TEMP,(real)1.3076e1)) + (real)6.173e-1*EXP(DIV(-TEMP,(real)2.078e3))
    + EXP(DIV((real)-5.093e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(89) = RF(89) * PCOR;
    RB(89) = RB(89) * PCOR;

    PR = RKLOW(10) * DIV(CTB_10, RF(93));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)4.675e-1*EXP(DIV(-TEMP,(real)1.51e2)) + (real)5.325e-1*EXP(DIV(-TEMP,(real)1.038e3))
    + EXP(DIV((real)-4.97e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(93) = RF(93) * PCOR;
    RB(93) = RB(93) * PCOR;

    PR = RKLOW(11) * DIV(CTB_114, RF(114));
    PCOR = DIV(PR, ((real)1.0 + PR));
    RF(114) = RF(114) * PCOR;
    RB(114) = RB(114) * PCOR;

    PR = RKLOW(12) * DIV(CTB_10, RF(115));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)-9.816e-1*EXP(DIV(-TEMP,(real)5.3837e3)) +
            (real)1.9816e0*EXP(DIV(-TEMP,(real)4.2932e0))  +
            EXP(DIV((real)7.95e-2,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(115) = RF(115) * PCOR;
    RB(115) = RB(115) * PCOR;
}
template<class real> void cpu_ratxb_kernel(const real* RESTRICT T, const real* RESTRICT C, real* RESTRICT RF,
        real* RESTRICT RB, const real* RESTRICT RKLOW, real TCONV){
    const real TEMP = T[0]*TCONV;
    
    real CTOT = (real)0.0;
    register real PR, PCOR, PRLOG, FCENT, FCLOG, XN;
    register real CPRLOG, FLOG, FC, SQR;
    const real SMALL = FLT_MIN;

    for (unsigned int k=1; k<=22; k++)
    {
        CTOT += C(k);
    }

    real CTB_5  = CTOT - C(1) - C(6) + C(10) - C(12) + (real)2.e0*C(16)
                + (real)2.e0*C(14) + (real)2.e0*C(15) ;
    real CTB_9  = CTOT - (real)2.7e-1*C(1) + (real)2.65e0*C(6) + C(10) + (real)2.e0*C(16)
                + (real)2.e0*C(14) + (real)2.e0*C(15) ;
    real CTB_10 = CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11) + C(12)
                + (real)2.e0*C(16) + (real)2.e0*C(14) + (real)2.e0*C(15);
    real CTB_11 = CTOT + (real)1.4e0*C(1) + (real)1.44e1*C(6) + C(10) + (real)7.5e-1*C(11)
                + (real)2.6e0*C(12) + (real)2.e0*C(16) + (real)2.e0*C(14)
                + (real)2.e0*C(15) ;
    real CTB_12 = CTOT - C(4) - C(6) - (real)2.5e-1*C(11) + (real)5.e-1*C(12)
                + (real)5.e-1*C(16) - C(22) + (real)2.e0*C(14) + (real)2.e0*C(15) ;
    real CTB_29 = CTOT + C(1) + (real)5.e0*C(4) + (real)5.e0*C(6) + C(10)
                + (real)5.e-1*C(11) + (real)2.5e0*C(12) + (real)2.e0*C(16)
                + (real)2.e0*C(14) + (real)2.e0*C(15) ;
    real CTB_190= CTOT + C(1) + (real)5.e0*C(6) + C(10) + (real)5.e-1*C(11)
                + C(12) + (real)2.e0*C(16) ;

    RF(5) = RF(5)*CTB_5*C(2)*C(2);
    RB(5) = RB(5)*CTB_5*C(1);
    RF(9) = RF(9)*CTB_9*C(2)*C(5);
    RB(9) = RB(9)*CTB_9*C(6);
    RF(10) = RF(10)*CTB_10*C(3)*C(2);
    RB(10) = RB(10)*CTB_10*C(5);
    RF(11) = RF(11)*CTB_11*C(3)*C(3);
    RB(11) = RB(11)*CTB_11*C(4);
    RF(12) = RF(12)*CTB_12*C(2)*C(4);
    RB(12) = RB(12)*CTB_12*C(7);
    RF(29) = RF(29)*CTB_29*C(11)*C(3);
    RB(29) = RB(29)*CTB_29*C(12);
    RF(46) = RF(46)*CTB_10;
    RB(46) = RB(46)*CTB_10*C(11)*C(2);
    RF(121) = RF(121)*CTOT*C(14)*C(9);
    RB(121) = RB(121)*CTOT*C(20);


    PR = RKLOW(13) * DIV(CTB_10, RF(126));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)6.63e-1*EXP(DIV(-TEMP,(real)1.707e3)) + (real)3.37e-1*EXP(DIV(-TEMP,(real)3.2e3))
    + EXP(DIV((real)-4.131e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(126) = RF(126) * PCOR;
    RB(126) = RB(126) * PCOR;

    PR = RKLOW(14) * DIV(CTB_10, RF(132));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.18e-1*EXP(DIV(-TEMP,(real)2.075e2)) + (real)7.82e-1*EXP(DIV(-TEMP,(real)2.663e3))
    + EXP(DIV((real)-6.095e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(132) = RF(132) * PCOR;
    RB(132) = RB(132) * PCOR;

    PR = RKLOW(15) * DIV(CTB_10, RF(145));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)8.25e-1*EXP(DIV(-TEMP,(real)1.3406e3)) + (real)1.75e-1*EXP(DIV(-TEMP,(real)6.e4))
    + EXP(DIV((real)-1.01398e4,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(145) = RF(145) * PCOR;
    RB(145) = RB(145) * PCOR;

    PR = RKLOW(16) * DIV(CTB_10, RF(148));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)4.5e-1*EXP(DIV(-TEMP,(real)8.9e3)) + (real)5.5e-1*EXP(DIV(-TEMP,(real)4.35e3))
    + EXP(DIV((real)-7.244e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(148) = RF(148) * PCOR;
    RB(148) = RB(148) * PCOR;

    PR = RKLOW(17) * DIV(CTB_10, RF(155));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.655e-1*EXP(DIV(-TEMP,(real)1.8e2)) + (real)7.345e-1*EXP(DIV(-TEMP,(real)1.035e3))
    + EXP(DIV((real)-5.417e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(155) = RF(155) * PCOR;
    RB(155) = RB(155) * PCOR;

    PR = RKLOW(18) * DIV(CTB_10, RF(156));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)2.47e-2*EXP(DIV(-TEMP,(real)2.1e2)) + (real)9.753e-1*EXP(DIV(-TEMP,(real)9.84e2))
    + EXP(DIV((real)-4.374e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(156) = RF(156) * PCOR;
    RB(156) = RB(156) * PCOR;

    PR = RKLOW(19) * DIV(CTB_10, RF(170));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)1.578e-1*EXP(DIV(-TEMP,(real)1.25e2)) + (real)8.422e-1*EXP(DIV(-TEMP,(real)2.219e3))
    + EXP(DIV((real)-6.882e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(170) = RF(170) * PCOR;
    RB(170) = RB(170) * PCOR;

    PR = RKLOW(20) * DIV(CTB_10, RF(185));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)9.8e-1*EXP(DIV(-TEMP,(real)1.0966e3)) + (real)2.e-2*EXP(DIV(-TEMP,(real)1.0966e3))
    + EXP(DIV((real)-6.8595e3,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(185) = RF(185) * PCOR;
    RB(185) = RB(185) * PCOR;

    PR = RKLOW(21) * DIV(CTB_190, RF(190));
    PCOR = DIV(PR, ((real)1.0 + PR));
    PRLOG = LOG10(MAX(PR,SMALL));
    FCENT = (real)0.e0*EXP(DIV(-TEMP,(real)1.e3)) + (real)1.e0*EXP(DIV(-TEMP,(real)1.31e3))
    + EXP(DIV((real)-4.8097e4,TEMP));
    FCLOG = LOG10(MAX(FCENT,SMALL));
    XN    = (real)0.75 - (real)1.27*FCLOG;
    CPRLOG= PRLOG - ((real)0.4 + (real)0.67*FCLOG);
    SQR = DIV(CPRLOG, (XN-(real)0.14*CPRLOG));
    FLOG = DIV(FCLOG, ((real)1.0 + SQR*SQR));
    FC = EXP10(FLOG);
    PCOR = FC * PCOR;
    RF(190) = RF(190) * PCOR;
    RB(190) = RB(190) * PCOR;

}
template<class real> void cpu_ratx2_kernel(const real* RESTRICT C, real* RESTRICT RF, real* RESTRICT RB){

    RF(1) = RF(1)*C(2)*C(4);
    RF(2) = RF(2)*C(3)*C(1);
    RF(3) = RF(3)*C(5)*C(1);
    RF(4) = RF(4)*C(5)*C(5);
    RF(6) = RF(6)*C(2)*C(2)*C(1);
    RF(7) = RF(7)*C(2)*C(2)*C(6);
    RF(8) = RF(8)*C(2)*C(2)*C(12);
    RF(13) = RF(13)*C(2)*C(4)*C(4);
    RF(14) = RF(14)*C(2)*C(4)*C(6);
    RF(15) = RF(15)*C(2)*C(4)*C(22);
    RF(16) = RF(16)*C(5)*C(5);
    RF(17) = RF(17)*C(7)*C(2);
    RF(18) = RF(18)*C(7)*C(2);
    RF(19) = RF(19)*C(7)*C(2);
    RF(20) = RF(20)*C(7)*C(3);
    RF(21) = RF(21)*C(7)*C(5);
    RF(22) = RF(22)*C(7)*C(7);
    RF(23) = RF(23)*C(7)*C(7);
    RF(24) = RF(24)*C(8)*C(2);
    RF(25) = RF(25)*C(8)*C(2);
    RF(26) = RF(26)*C(8)*C(3);
    RF(27) = RF(27)*C(8)*C(5);
    RF(28) = RF(28)*C(8)*C(5);
    RF(30) = RF(30)*C(11)*C(5);
    RF(31) = RF(31)*C(11)*C(1);
    RF(32) = RF(32)*C(11)*C(4);
    RF(33) = RF(33)*C(11)*C(7);
    RF(34) = RF(34)*C(3);
    RF(35) = RF(35)*C(5);
    RF(36) = RF(36)*C(1);
    RF(37) = RF(37)*C(6);
    RF(38) = RF(38)*C(4);
    RF(39) = RF(39)*C(11);
    RF(40) = RF(40)*C(12);
    RF(41) = RF(41)*C(2);
    RF(42) = RF(42)*C(2);
    RF(43) = RF(43)*C(3);
    RF(44) = RF(44)*C(3);
    RF(45) = RF(45)*C(5);
    RF(47) = RF(47)*C(4);
    RF(48) = RF(48)*C(2);
    RF(49) = RF(49)*C(1);
    RF(50) = RF(50)*C(3);
    RF(51) = RF(51)*C(4);
    RF(52) = RF(52)*C(4);
    RF(53) = RF(53)*C(5);
    RF(54) = RF(54)*C(5);
    RF(55) = RF(55)*C(7);
    RF(56) = RF(56)*C(11);
    RF(59) = RF(59)*C(22);
    RF(60) = RF(60)*C(2);
    RF(61) = RF(61)*C(3);
    RF(62) = RF(62)*C(3);
    RF(63) = RF(63)*C(5);
    RF(64) = RF(64)*C(1);
    RF(65) = RF(65)*C(4);
    RF(66) = RF(66)*C(4);
    RF(67) = RF(67)*C(6);
    RF(68) = RF(68)*C(11);
    RF(69) = RF(69)*C(12);
    RF(70) = RF(70)*C(12);
    RF(71) = RF(71)*C(13)*C(2);
    RF(72) = RF(72)*C(13)*C(2);
    RF(73) = RF(73)*C(13)*C(3);
    RF(74) = RF(74)*C(13)*C(5);
    RF(75) = RF(75)*C(13)*C(4);
    RF(76) = RF(76)*C(13)*C(7);
    RF(77) = RF(77)*C(13);
    RF(78) = RF(78)*C(9)*C(2);
    RF(79) = RF(79)*C(9)*C(3);
    RF(80) = RF(80)*C(9)*C(5);
    RF(81) = RF(81)*C(9)*C(5);
    RF(82) = RF(82)*C(9)*C(4);
    RF(83) = RF(83)*C(9)*C(4);
    RF(84) = RF(84)*C(9)*C(7);
    RF(85) = RF(85)*C(9)*C(7);
    RF(86) = RF(86)*C(9)*C(8);
    RF(87) = RF(87)*C(9);
    RF(88) = RF(88)*C(9);
    RF(89) = RF(89)*C(9);
    RF(90) = RF(90)*C(9)*C(13);
    RF(91) = RF(91)*C(9);
    RF(92) = RF(92)*C(9);
    RF(93) = RF(93)*C(9)*C(9);
    RF(94) = RF(94)*C(9)*C(9);
    RF(95) = RF(95)*C(9)*C(17);
    RF(96) = RF(96)*C(2);
    RF(97) = RF(97)*C(2);
    RF(98) = RF(98)*C(2);
    RF(99) = RF(99)*C(3);
    RF(100) = RF(100)*C(5);
    RF(101) = RF(101)*C(4);
    RF(102) = RF(102)*C(10)*C(2);
    RF(103) = RF(103)*C(10)*C(3);
    RF(104) = RF(104)*C(10)*C(5);
    RF(105) = RF(105)*C(10);
    RF(106) = RF(106)*C(10);
    RF(107) = RF(107)*C(10);
    RF(108) = RF(108)*C(17)*C(2);
    RF(109) = RF(109)*C(17)*C(3);
    RF(110) = RF(110)*C(17)*C(4);
    RF(111) = RF(111)*C(17);
    RF(112) = RF(112)*C(17);
    RF(113) = RF(113)*C(17)*C(17);
    RF(114) = RF(114)*C(14);
    RF(116) = RF(116)*C(14)*C(3);
    RF(117) = RF(117)*C(14)*C(3);
    RF(118) = RF(118)*C(14)*C(5);
    RF(119) = RF(119)*C(14)*C(5);
    RF(120) = RF(120)*C(14);
    RF(122) = RF(122)*C(2);
    RF(123) = RF(123)*C(3);
    RF(124) = RF(124)*C(5);
    RF(125) = RF(125)*C(4);
    RF(126) = RF(126)*C(18)*C(2);
    RF(127) = RF(127)*C(18)*C(2);
    RF(128) = RF(128)*C(18)*C(2);
    RF(129) = RF(129)*C(18)*C(3);
    RF(130) = RF(130)*C(18)*C(3);
    RF(131) = RF(131)*C(18)*C(5);
    RF(132) = RF(132)*C(2);
    RF(133) = RF(133)*C(2);
    RF(134) = RF(134)*C(2);
    RF(135) = RF(135)*C(3);
    RF(136) = RF(136)*C(3);
    RF(137) = RF(137)*C(5);
    RF(138) = RF(138)*C(4);
    RF(139) = RF(139)*C(4);
    RF(140) = RF(140)*C(4);
    RF(141) = RF(141)*C(7);
    RF(142) = RF(142)*C(8);
    RF(144) = RF(144)*C(9);
    RF(145) = RF(145)*C(9);
    RF(146) = RF(146)*C(9);
    RF(148) = RF(148)*C(2);
    RF(149) = RF(149)*C(2);
    RF(150) = RF(150)*C(2);
    RF(151) = RF(151)*C(3);
    RF(152) = RF(152)*C(5);
    RF(153) = RF(153)*C(4);
    RF(154) = RF(154)*C(4);
    RF(155) = RF(155)*C(15);
    RF(156) = RF(156)*C(15)*C(2);
    RF(157) = RF(157)*C(15)*C(2);
    RF(158) = RF(158)*C(15)*C(3);
    RF(159) = RF(159)*C(15)*C(3);
    RF(160) = RF(160)*C(15)*C(3);
    RF(161) = RF(161)*C(15)*C(5);
    RF(162) = RF(162)*C(15)*C(4);
    RF(163) = RF(163)*C(15)*C(7);
    RF(164) = RF(164)*C(15);
    RF(165) = RF(165)*C(15);
    RF(166) = RF(166)*C(15);
    RF(167) = RF(167)*C(15);
    RF(168) = RF(168)*C(15)*C(9);
    RF(169) = RF(169)*C(15)*C(9);
    RF(170) = RF(170)*C(2);
    RF(171) = RF(171)*C(2);
    RF(172) = RF(172)*C(3);
    RF(173) = RF(173)*C(3);
    RF(174) = RF(174)*C(4);
    RF(175) = RF(175)*C(7);
    RF(176) = RF(176)*C(7);
    RF(177) = RF(177)*C(7);
    RF(178) = RF(178)*C(8);
    RF(180) = RF(180)*C(16)*C(2);
    RF(181) = RF(181)*C(16)*C(3);
    RF(182) = RF(182)*C(16)*C(5);
    RF(183) = RF(183)*C(16);
    RF(184) = RF(184)*C(16)*C(9);
    RF(185) = RF(185)*C(20)*C(2);
    RF(186) = RF(186)*C(20)*C(2);
    RF(187) = RF(187)*C(20)*C(7);
    RF(188) = RF(188)*C(20)*C(7);
    RF(189) = RF(189)*C(20);
    RF(190) = RF(190)*C(21)*C(2);
    RF(191) = RF(191)*C(21)*C(2);
    RF(192) = RF(192)*C(21)*C(2);
    RF(193) = RF(193)*C(21)*C(3);
    RF(194) = RF(194)*C(21)*C(3);
    RF(195) = RF(195)*C(21)*C(3);
    RF(196) = RF(196)*C(21)*C(5);
    RF(197) = RF(197)*C(21)*C(7);
    RF(198) = RF(198)*C(21)*C(9);
    RF(199) = RF(199)*C(2);
    RF(200) = RF(200)*C(2);
    RF(201) = RF(201)*C(3);
    RF(202) = RF(202)*C(5);
    RF(203) = RF(203)*C(4);
    RF(204) = RF(204)*C(7);
    RF(205) = RF(205)*C(9);
}
template<class real> void cpu_ratx4_kernel(const real* RESTRICT C, real* RESTRICT RF, real* RESTRICT RB){
    RB(1) = RB(1)*C(3)*C(5);
    RB(2) = RB(2)*C(2)*C(5);
    RB(3) = RB(3)*C(2)*C(6);
    RB(4) = RB(4)*C(3)*C(6);
    RB(6) = RB(6)*C(1)*C(1);
    RB(7) = RB(7)*C(1)*C(6);
    RB(8) = RB(8)*C(1)*C(12);
    RB(13) = RB(13)*C(7)*C(4);
    RB(14) = RB(14)*C(7)*C(6);
    RB(15) = RB(15)*C(7)*C(22);
    RB(16) = RB(16)*C(8);
    RB(17) = RB(17)*C(3)*C(6);
    RB(18) = RB(18)*C(4)*C(1);
    RB(19) = RB(19)*C(5)*C(5);
    RB(20) = RB(20)*C(5)*C(4);
    RB(21) = RB(21)*C(4)*C(6);
    RB(22) = RB(22)*C(4)*C(8);
    RB(23) = RB(23)*C(4)*C(8);
    RB(24) = RB(24)*C(7)*C(1);
    RB(25) = RB(25)*C(5)*C(6);
    RB(26) = RB(26)*C(5)*C(7);
    RB(27) = RB(27)*C(7)*C(6);
    RB(28) = RB(28)*C(7)*C(6);
    RB(30) = RB(30)*C(12)*C(2);
    RB(31) = RB(31)*C(13);
    RB(32) = RB(32)*C(12)*C(3);
    RB(33) = RB(33)*C(12)*C(5);
    RB(34) = RB(34)*C(11)*C(2);
    RB(35) = RB(35)*C(2);
    RB(36) = RB(36)*C(2);
    RB(37) = RB(37)*C(13)*C(2);
    RB(38) = RB(38)*C(3);
    RB(39) = RB(39)*C(17);
    RB(40) = RB(40)*C(11);
    RB(41) = RB(41)*C(13);
    RB(42) = RB(42)*C(11)*C(1);
    RB(43) = RB(43)*C(11)*C(5);
    RB(44) = RB(44)*C(12)*C(2);
    RB(45) = RB(45)*C(11)*C(6);
    RB(47) = RB(47)*C(11)*C(7);
    RB(48) = RB(48)*C(9);
    RB(49) = RB(49)*C(2)*C(9);
    RB(50) = RB(50)*C(2);
    RB(51) = RB(51)*C(5);
    RB(52) = RB(52)*C(12)*C(2)*C(2);
    RB(53) = RB(53)*C(13)*C(2);
    RB(54) = RB(54)*C(6);
    RB(55) = RB(55)*C(13)*C(5);
    RB(56) = RB(56)*C(18);
    RB(57) = RB(57)*C(14)*C(2);
    RB(58) = RB(58)*C(14)*C(1);
    RB(59) = RB(59)*C(22);
    RB(60) = RB(60)*C(1);
    RB(61) = RB(61)*C(11)*C(1);
    RB(62) = RB(62)*C(2);
    RB(63) = RB(63)*C(13)*C(2);
    RB(64) = RB(64)*C(9)*C(2);
    RB(65) = RB(65)*C(2)*C(5)*C(11);
    RB(66) = RB(66)*C(11)*C(6);
    RB(67) = RB(67)*C(6);
    RB(68) = RB(68)*C(11);
    RB(69) = RB(69)*C(12);
    RB(70) = RB(70)*C(13)*C(11);
    RB(72) = RB(72)*C(1);
    RB(73) = RB(73)*C(5);
    RB(74) = RB(74)*C(6);
    RB(75) = RB(75)*C(7);
    RB(76) = RB(76)*C(8);
    RB(77) = RB(77)*C(18)*C(2);
    RB(78) = RB(78)*C(10);
    RB(79) = RB(79)*C(13)*C(2);
    RB(80) = RB(80)*C(6);
    RB(81) = RB(81)*C(6);
    RB(82) = RB(82)*C(3);
    RB(83) = RB(83)*C(5)*C(13);
    RB(84) = RB(84)*C(10)*C(4);
    RB(85) = RB(85)*C(5);
    RB(86) = RB(86)*C(10)*C(7);
    RB(87) = RB(87)*C(2);
    RB(88) = RB(88)*C(10)*C(11);
    RB(89) = RB(89)*C(19);
    RB(90) = RB(90)*C(10);
    RB(91) = RB(91)*C(15)*C(2);
    RB(92) = RB(92)*C(15)*C(2);
    RB(93) = RB(93)*C(16);
    RB(94) = RB(94)*C(2);
    RB(95) = RB(95)*C(15)*C(11);
    RB(96) = RB(96)*C(13)*C(1);
    RB(97) = RB(97)*C(9)*C(5);
    RB(98) = RB(98)*C(6);
    RB(99) = RB(99)*C(13)*C(5);
    RB(100) = RB(100)*C(13)*C(6);
    RB(101) = RB(101)*C(13)*C(7);
    RB(102) = RB(102)*C(9)*C(1);
    RB(103) = RB(103)*C(9)*C(5);
    RB(104) = RB(104)*C(9)*C(6);
    RB(105) = RB(105)*C(15)*C(2);
    RB(106) = RB(106)*C(9)*C(9);
    RB(107) = RB(107)*C(9)*C(9);
    RB(108) = RB(108)*C(11);
    RB(109) = RB(109)*C(2)*C(11)*C(11);
    RB(110) = RB(110)*C(5)*C(11)*C(11);
    RB(111) = RB(111)*C(14)*C(11);
    RB(112) = RB(112)*C(11);
    RB(113) = RB(113)*C(14)*C(11)*C(11);
    RB(115) = RB(115)*C(14)*C(2);
    RB(116) = RB(116)*C(17)*C(2);
    RB(117) = RB(117)*C(11);
    RB(118) = RB(118)*C(18)*C(2);
    RB(119) = RB(119)*C(9)*C(11);
    RB(120) = RB(120)*C(11);
    RB(122) = RB(122)*C(14)*C(2);
    RB(123) = RB(123)*C(11);
    RB(124) = RB(124)*C(18)*C(2);
    RB(125) = RB(125)*C(12);
    RB(127) = RB(127)*C(17)*C(1);
    RB(128) = RB(128)*C(9)*C(11);
    RB(129) = RB(129)*C(17)*C(5);
    RB(130) = RB(130)*C(12);
    RB(131) = RB(131)*C(17)*C(6);
    RB(132) = RB(132)*C(15);
    RB(133) = RB(133)*C(14)*C(1);
    RB(134) = RB(134)*C(1);
    RB(135) = RB(135)*C(18)*C(2);
    RB(136) = RB(136)*C(9)*C(11);
    RB(137) = RB(137)*C(14)*C(6);
    RB(138) = RB(138)*C(14)*C(7);
    RB(139) = RB(139)*C(3);
    RB(140) = RB(140)*C(13);
    RB(141) = RB(141)*C(5);
    RB(142) = RB(142)*C(15)*C(7);
    RB(143) = RB(143)*C(15)*C(11);
    RB(144) = RB(144)*C(14)*C(10);
    RB(145) = RB(145)*C(21);
    RB(146) = RB(146)*C(20)*C(2);
    RB(147) = RB(147)*C(9)*C(11);
    RB(148) = RB(148)*C(19);
    RB(149) = RB(149)*C(9);
    RB(150) = RB(150)*C(18)*C(1);
    RB(151) = RB(151)*C(18)*C(5);
    RB(152) = RB(152)*C(18)*C(6);
    RB(153) = RB(153)*C(18)*C(7);
    RB(154) = RB(154)*C(13)*C(11)*C(5);
    RB(155) = RB(155)*C(1);
    RB(157) = RB(157)*C(1);
    RB(158) = RB(158)*C(5);
    RB(159) = RB(159)*C(9);
    RB(160) = RB(160)*C(13);
    RB(161) = RB(161)*C(6);
    RB(162) = RB(162)*C(7);
    RB(163) = RB(163)*C(19)*C(5);
    RB(164) = RB(164)*C(11);
    RB(165) = RB(165)*C(20)*C(2);
    RB(166) = RB(166)*C(10);
    RB(167) = RB(167)*C(20)*C(2);
    RB(168) = RB(168)*C(10);
    RB(170) = RB(170)*C(16);
    RB(171) = RB(171)*C(15)*C(1);
    RB(172) = RB(172)*C(9)*C(13);
    RB(173) = RB(173)*C(19)*C(2);
    RB(174) = RB(174)*C(15)*C(7);
    RB(175) = RB(175)*C(16)*C(4);
    RB(176) = RB(176)*C(15)*C(8);
    RB(177) = RB(177)*C(9)*C(13)*C(5);
    RB(178) = RB(178)*C(16)*C(7);
    RB(179) = RB(179)*C(16)*C(11);
    RB(180) = RB(180)*C(1);
    RB(181) = RB(181)*C(5);
    RB(182) = RB(182)*C(6);
    RB(183) = RB(183)*C(9);
    RB(184) = RB(184)*C(10);
    RB(185) = RB(185)*C(21);
    RB(186) = RB(186)*C(10);
    RB(187) = RB(187)*C(21)*C(4);
    RB(188) = RB(188)*C(5)*C(13);
    RB(189) = RB(189)*C(21)*C(11);
    RB(191) = RB(191)*C(15)*C(9);
    RB(192) = RB(192)*C(20)*C(1);
    RB(193) = RB(193)*C(18)*C(9)*C(2);
    RB(195) = RB(195)*C(20)*C(5);
    RB(196) = RB(196)*C(20)*C(6);
    RB(197) = RB(197)*C(20)*C(8);
    RB(198) = RB(198)*C(20)*C(10);
    RB(199) = RB(199)*C(9);
    RB(200) = RB(200)*C(21)*C(1);
    RB(201) = RB(201)*C(13);
    RB(202) = RB(202)*C(21)*C(6);
    RB(203) = RB(203)*C(21)*C(7);
    RB(204) = RB(204)*C(5)*C(13);
    RB(205) = RB(205)*C(10)*C(21);
    RB(206) = RB(206)*C(20)*C(9);
}
template<class real> void cpu_qssa_kernel(real* RESTRICT RF, real* RESTRICT RB, real* RESTRICT A){

    register real DEN;

    RF(57)  = (real)0.0;
    RF(58)  = (real)0.0;
    RF(143) = (real)0.0;
    RF(179) = (real)0.0;
    RB(194) = (real)0.0;
    RF(206) = (real)0.0;

    
    DEN = +RF(34) +RF(35) +RF(36) +RF(37) +RF(38)
               +RF(39) +RF(40) +RF(77) +RF(87) +RF(105) +RF(111)
               +RB(54) +RB(60) ;
    A(1,0) = DIV ((+RB(34) +RB(37) +RB(39) +RB(57) +RB(77)
            +RB(105) +RB(111)), DEN);
    A(1,2) = DIV ((+RB(36) +RF(54)), DEN);
    A(1,3) = DIV ((+RF(60)), DEN);
    A(1,4) = DIV ((+RB(35) +RB(38) +RB(40)), DEN);
    A(1,7) = DIV ((+RB(87)), DEN);
    
    DEN = +RF(48) +RF(49) +RF(50) +RF(51) +RF(52)
               +RF(53) +RF(54) +RF(55) +RF(56) +RF(91) +RF(106)
               +RF(112) +RF(165) +RB(36) +RB(59) +RB(67) +RB(68)
               +RB(69) +RB(80) +RB(117) +RB(123) +RB(125) +RB(130)
               +RB(160) ;
    A(2,0) = DIV ((+RB(48) +RB(49) +RB(52) +RB(53) +RB(55)
            +RB(56) +RB(57) +RB(58) +RB(58) +RF(80) +RB(91)
            +RB(106) +RF(117) +RF(130) +RF(160) +RB(165)), DEN);
    A(2,1) = DIV ((+RF(36) +RB(54)), DEN);
    A(2,3) = DIV ((+RF(59) +RF(67) +RF(68) +RF(69)), DEN);
    A(2,4) = DIV ((+RB(50) +RB(51)), DEN);
    A(2,6) = DIV ((+RF(123) +RF(125)), DEN);
    A(2,7) = DIV ((+RB(112)), DEN);
    
    DEN = +RF(59) +RF(60) +RF(61) +RF(62) +RF(63)
               +RF(64) +RF(65) +RF(66) +RF(67) +RF(68) +RF(69)
               +RF(70) +RF(92) +RF(107) +RF(166) +RF(167) +RF(183)
               +RB(81) +RB(98) +RB(108) ;
    A(3,0) = DIV ((+RB(61) +RB(63) +RB(64) +RB(65) +RB(66)
            +RB(70) +RF(81) +RB(92) +RB(107) +RF(108) +RB(167)), DEN);
    A(3,1) = DIV ((+RB(60)), DEN);
    A(3,2) = DIV ((+RB(59) +RB(67) +RB(68) +RB(69)), DEN);
    A(3,4) = DIV ((+RB(62)), DEN);
    A(3,5) = DIV ((+RF(98)), DEN);
    A(3,6) = DIV ((+RB(166)), DEN);
    A(3,8) = DIV ((+RB(183)), DEN);
    
    DEN = +RF(41) +RF(42) +RF(43) +RF(44) +RF(45)
               +RF(46) +RF(47) +RF(88) +RF(89) +RF(120) +RF(164)
               +RF(189) +RB(35) +RB(38) +RB(40) +RB(50) +RB(51)
               +RB(62) +RB(72) +RB(73) +RB(74) +RB(75) +RB(76)
               +RB(90) +RB(140) +RB(149) +RB(159) ;
    A(4,0) = DIV ((+RB(41) +RB(42) +RB(43) +RB(44) +RB(45)
            +RB(46) +RB(47) +RF(72) +RF(73) +RF(74) +RF(75)
            +RF(76) +RB(88) +RB(89) +RF(90) +RB(143) +RF(159)
            +RB(179) +RB(189) +RF(194)), DEN);
    A(4,1) = DIV ((+RF(35) +RF(38) +RF(40)), DEN);
    A(4,2) = DIV ((+RF(50) +RF(51)), DEN);
    A(4,3) = DIV ((+RF(62)), DEN);
    A(4,7) = DIV ((+RB(120) +RF(140)), DEN);
    A(4,8) = DIV ((+RB(164)), DEN);
    A(4,9) = DIV ((+RF(149)), DEN);
    
    DEN = +RF(96) +RF(97) +RF(98) +RF(99) +RF(100)
               +RF(101) +RB(71) +RB(82) +RB(85) ;
    A(5,0) = DIV ((+RF(71) +RF(82) +RF(85) +RB(96) +RB(97)
            +RB(99) +RB(100) +RB(101)), DEN);
    A(5,3) = DIV((+RB(98)), DEN);
    
    DEN = +RF(122) +RF(123) +RF(124) +RF(125) +RB(114)
               +RB(134) +RB(155) +RB(166) +RB(186) ;
    A(6,0) = DIV ((+RF(114) +RB(122) +RB(124) +RF(155) +RF(186)), DEN);
    A(6,2) = DIV ((+RB(123) +RB(125)), DEN);
    A(6,3) = DIV ((+RF(166)), DEN);
    A(6,7) = DIV ((+RF(134)), DEN);
    
    DEN = +RF(115) +RF(132) +RF(133) +RF(134) +RF(135)
               +RF(136) +RF(137) +RF(138) +RF(139) +RF(140) +RF(141)
               +RF(142) +RF(144) +RF(145) +RF(146) +RB(87) +RB(112)
               +RB(120) +RB(157) +RB(158) +RB(161) +RB(162) +RB(168)
               +RB(188) ;
    A(7,0) = DIV ((+RB(115) +RB(132) +RB(133) +RB(135) +RB(136)
            +RB(137) +RB(138) +RB(142) +RB(143) +RB(144) +RB(145)
            +RB(146) +RF(157) +RF(158) +RF(161) +RF(162) +RF(168)
            +RF(188) +RB(206)), DEN);
    A(7,1) = DIV ((+RF(87)), DEN);
    A(7,2) = DIV ((+RF(112)), DEN);
    A(7,4) = DIV ((+RF(120) +RB(140)), DEN);
    A(7,6) = DIV ((+RB(134)), DEN);
    A(7,9) = DIV ((+RB(139) +RB(141)), DEN);
    
    DEN = +RF(170) +RF(171) +RF(172) +RF(173) +RF(174)
               +RF(175) +RF(176) +RF(177) +RF(178) +RB(94) +RB(156)
               +RB(164) +RB(180) +RB(181) +RB(182) +RB(183) +RB(184)
               +RB(199) +RB(201) +RB(204) ;
    A(8,0) = DIV ((+RF(94) +RF(156) +RB(170) +RB(171) +RB(172)
            +RB(173) +RB(174) +RB(175) +RB(176) +RB(177) +RB(178)
            +RB(179) +RF(180) +RF(181) +RF(182) +RF(184) +RF(194)
            +RB(206)), DEN);
    A(8,3) = DIV ((+RF(183)), DEN);
    A(8,4) = DIV ((+RF(164)), DEN);
    A(8,10) = DIV ((+RF(199) +RF(201) +RF(204)), DEN);
    
    DEN = +RF(147) +RF(148) +RF(149) +RF(150) +RF(151)
               +RF(152) +RF(153) +RF(154) +RB(126) +RB(139) +RB(141) ;
    A(9,0) = DIV ((+RF(126) +RB(147) +RB(148) +RB(150) +RB(151)
            +RB(152) +RB(153) +RB(154)), DEN);
    A(9,4) = DIV ((+RB(149)), DEN);
    A(9,7) = DIV ((+RF(139) +RF(141)), DEN);

    DEN = +RF(199) +RF(200) +RF(201) +RF(202) +RF(203)
               +RF(204) +RF(205) +RB(169) +RB(190) ;
    A(10,0) = DIV ((+RF(169) +RF(190) +RB(200) +RB(202) +RB(203)
            +RB(205)), DEN);
    A(10,8) = DIV ((+RB(199) +RB(201) +RB(204)), DEN);
}
template<class real> void cpu_qssab_kernel(real* RESTRICT RF, real* RESTRICT RB, real* RESTRICT A){

    register real DEN;

    A(8,0) = A(8,0) + A(8,10)*A(10,0);
    DEN = 1 -A(8,10)*A(10,8);
    A(8,0) = DIV (A(8,0), DEN);
    A(8,4) = DIV (A(8,4), DEN);
    A(8,3) = DIV (A(8,3), DEN);
    A(3,0) = A(3,0) + A(3,5)*A(5,0);
    DEN = 1 -A(3,5)*A(5,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,2) = DIV (A(3,2), DEN);
    A(3,1) = DIV (A(3,1), DEN);
    A(3,8) = DIV (A(3,8), DEN);
    A(3,6) = DIV (A(3,6), DEN);
    A(4,0) = A(4,0) + A(4,9)*A(9,0);
    A(4,7) = A(4,7) + A(4,9)*A(9,7);
    DEN = 1 -A(4,9)*A(9,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(4,7) = DIV (A(4,7), DEN);
    A(4,2) = DIV (A(4,2), DEN);
    A(4,1) = DIV (A(4,1), DEN);
    A(4,8) = DIV (A(4,8), DEN);
    A(7,0) = A(7,0) + A(7,9)*A(9,0);
    A(7,4) = A(7,4) + A(7,9)*A(9,4);
    DEN = 1 -A(7,9)*A(9,7);
    A(7,0) = DIV (A(7,0), DEN);
    A(7,4) = DIV (A(7,4), DEN);
    A(7,2) = DIV (A(7,2), DEN);
    A(7,1) = DIV (A(7,1), DEN);
    A(7,6) = DIV (A(7,6), DEN);
    A(3,0) = A(3,0) + A(3,6)*A(6,0);
    A(3,7) = A(3,6)*A(6,7);
    A(3,2) = A(3,2) + A(3,6)*A(6,2);
    DEN = 1 -A(3,6)*A(6,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,7) = DIV (A(3,7), DEN);
    A(3,2) = DIV (A(3,2), DEN);
    A(3,1) = DIV (A(3,1), DEN);
    A(3,8) = DIV (A(3,8), DEN);
    A(7,0) = A(7,0) + A(7,6)*A(6,0);
    A(7,3) = A(7,6)*A(6,3);
    A(7,2) = A(7,2) + A(7,6)*A(6,2);
    DEN = 1 -A(7,6)*A(6,7);
    A(7,0) = DIV (A(7,0), DEN);
    A(7,4) = DIV (A(7,4), DEN);
    A(7,3) = DIV (A(7,3), DEN);
    A(7,2) = DIV (A(7,2), DEN);
    A(7,1) = DIV (A(7,1), DEN);
    A(2,0) = A(2,0) + A(2,6)*A(6,0);
    A(2,3) = A(2,3) + A(2,6)*A(6,3);
    A(2,7) = A(2,7) + A(2,6)*A(6,7);
    DEN = 1 -A(2,6)*A(6,2);
    A(2,0) = DIV (A(2,0), DEN);
    A(2,4) = DIV (A(2,4), DEN);
    A(2,3) = DIV (A(2,3), DEN);
    A(2,7) = DIV (A(2,7), DEN);
    A(2,1) = DIV (A(2,1), DEN);
    A(4,0) = A(4,0) + A(4,8)*A(8,0);
    A(4,3) = A(4,3) + A(4,8)*A(8,3);
    DEN = 1 -A(4,8)*A(8,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(4,7) = DIV (A(4,7), DEN);
    A(4,2) = DIV (A(4,2), DEN);
    A(4,1) = DIV (A(4,1), DEN);
    A(3,0) = A(3,0) + A(3,8)*A(8,0);
    A(3,4) = A(3,4) + A(3,8)*A(8,4);
    DEN = 1 -A(3,8)*A(8,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,7) = DIV (A(3,7), DEN);
    A(3,2) = DIV (A(3,2), DEN);
    A(3,1) = DIV (A(3,1), DEN);
    A(4,0) = A(4,0) + A(4,1)*A(1,0);
    A(4,3) = A(4,3) + A(4,1)*A(1,3);
    A(4,7) = A(4,7) + A(4,1)*A(1,7);
    A(4,2) = A(4,2) + A(4,1)*A(1,2);
    DEN = 1 -A(4,1)*A(1,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(4,7) = DIV (A(4,7), DEN);
    A(4,2) = DIV (A(4,2), DEN);
    A(3,0) = A(3,0) + A(3,1)*A(1,0);
    A(3,4) = A(3,4) + A(3,1)*A(1,4);
    A(3,7) = A(3,7) + A(3,1)*A(1,7);
    A(3,2) = A(3,2) + A(3,1)*A(1,2);
    DEN = 1 -A(3,1)*A(1,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,7) = DIV (A(3,7), DEN);
    A(3,2) = DIV (A(3,2), DEN);
    A(7,0) = A(7,0) + A(7,1)*A(1,0);
    A(7,4) = A(7,4) + A(7,1)*A(1,4);
    A(7,3) = A(7,3) + A(7,1)*A(1,3);
    A(7,2) = A(7,2) + A(7,1)*A(1,2);
    DEN = 1 -A(7,1)*A(1,7);
    A(7,0) = DIV (A(7,0), DEN);
    A(7,4) = DIV (A(7,4), DEN);
    A(7,3) = DIV (A(7,3), DEN);
    A(7,2) = DIV (A(7,2), DEN);
    A(2,0) = A(2,0) + A(2,1)*A(1,0);
    A(2,4) = A(2,4) + A(2,1)*A(1,4);
    A(2,3) = A(2,3) + A(2,1)*A(1,3);
    A(2,7) = A(2,7) + A(2,1)*A(1,7);
    DEN = 1 -A(2,1)*A(1,2);
    A(2,0) = DIV (A(2,0), DEN);
    A(2,4) = DIV (A(2,4), DEN);
    A(2,3) = DIV (A(2,3), DEN);
    A(2,7) = DIV (A(2,7), DEN);
    A(4,0) = A(4,0) + A(4,2)*A(2,0);
    A(4,3) = A(4,3) + A(4,2)*A(2,3);
    A(4,7) = A(4,7) + A(4,2)*A(2,7);
    DEN = 1 -A(4,2)*A(2,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(4,7) = DIV (A(4,7), DEN);
    A(3,0) = A(3,0) + A(3,2)*A(2,0);
    A(3,4) = A(3,4) + A(3,2)*A(2,4);
    A(3,7) = A(3,7) + A(3,2)*A(2,7);
    DEN = 1 -A(3,2)*A(2,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(3,7) = DIV (A(3,7), DEN);
    A(7,0) = A(7,0) + A(7,2)*A(2,0);
    A(7,4) = A(7,4) + A(7,2)*A(2,4);
    A(7,3) = A(7,3) + A(7,2)*A(2,3);
    DEN = 1 -A(7,2)*A(2,7);
    A(7,0) = DIV (A(7,0), DEN);
    A(7,4) = DIV (A(7,4), DEN);
    A(7,3) = DIV (A(7,3), DEN);
    A(4,0) = A(4,0) + A(4,7)*A(7,0);
    A(4,3) = A(4,3) + A(4,7)*A(7,3);
    DEN = 1 -A(4,7)*A(7,4);
    A(4,0) = DIV (A(4,0), DEN);
    A(4,3) = DIV (A(4,3), DEN);
    A(3,0) = A(3,0) + A(3,7)*A(7,0);
    A(3,4) = A(3,4) + A(3,7)*A(7,4);
    DEN = 1 -A(3,7)*A(7,3);
    A(3,0) = DIV (A(3,0), DEN);
    A(3,4) = DIV (A(3,4), DEN);
    A(4,0) = A(4,0) + A(4,3)*A(3,0);
    DEN = 1 -A(4,3)*A(3,4);
    A(4,0) = DIV (A(4,0), DEN);
}
template<class real> void cpu_qssa2_kernel(real* RESTRICT RF, real* RESTRICT RB, const real* RESTRICT A){

    register real xq4 = A(4,0);
    register real xq3 = A(3,0) +A(3,4)*xq4;
    register real xq7 = A(7,0) +A(7,4)*xq4 +A(7,3)*xq3;
    register real xq2 = A(2,0) +A(2,4)*xq4 +A(2,3)*xq3 +A(2,7)*xq7;
    register real xq1 = A(1,0) +A(1,4)*xq4 +A(1,3)*xq3 +A(1,7)*xq7 +A(1,2)*xq2;
    register real xq8 = A(8,0) +A(8,4)*xq4 +A(8,3)*xq3;
    register real xq6 = A(6,0) +A(6,3)*xq3 +A(6,7)*xq7 +A(6,2)*xq2;
    register real xq9 = A(9,0) +A(9,4)*xq4 +A(9,7)*xq7;
    register real xq5 = A(5,0) +A(5,3)*xq3;
    register real xq10 = A(10,0) +A(10,8)*xq8;

    RF(34) = RF(34)*xq1;
    RF(35) = RF(35)*xq1;
    RB(35) = RB(35)*xq4;
    RF(36) = RF(36)*xq1;
    RB(36) = RB(36)*xq2;
    RF(37) = RF(37)*xq1;
    RF(38) = RF(38)*xq1;
    RB(38) = RB(38)*xq4;
    RF(39) = RF(39)*xq1;
    RF(40) = RF(40)*xq1;
    RB(40) = RB(40)*xq4;
    RF(41) = RF(41)*xq4;
    RF(42) = RF(42)*xq4;
    RF(43) = RF(43)*xq4;
    RF(44) = RF(44)*xq4;
    RF(45) = RF(45)*xq4;
    RF(46) = RF(46)*xq4;
    RF(47) = RF(47)*xq4;
    RF(48) = RF(48)*xq2;
    RF(49) = RF(49)*xq2;
    RF(50) = RF(50)*xq2;
    RB(50) = RB(50)*xq4;
    RF(51) = RF(51)*xq2;
    RB(51) = RB(51)*xq4;
    RF(52) = RF(52)*xq2;
    RF(53) = RF(53)*xq2;
    RF(54) = RF(54)*xq2;
    RB(54) = RB(54)*xq1;
    RF(55) = RF(55)*xq2;
    RF(56) = RF(56)*xq2;
    RF(59) = RF(59)*xq3;
    RB(59) = RB(59)*xq2;
    RF(60) = RF(60)*xq3;
    RB(60) = RB(60)*xq1;
    RF(61) = RF(61)*xq3;
    RF(62) = RF(62)*xq3;
    RB(62) = RB(62)*xq4;
    RF(63) = RF(63)*xq3;
    RF(64) = RF(64)*xq3;
    RF(65) = RF(65)*xq3;
    RF(66) = RF(66)*xq3;
    RF(67) = RF(67)*xq3;
    RB(67) = RB(67)*xq2;
    RF(68) = RF(68)*xq3;
    RB(68) = RB(68)*xq2;
    RF(69) = RF(69)*xq3;
    RB(69) = RB(69)*xq2;
    RF(70) = RF(70)*xq3;
    RB(71) = RB(71)*xq5;
    RB(72) = RB(72)*xq4;
    RB(73) = RB(73)*xq4;
    RB(74) = RB(74)*xq4;
    RB(75) = RB(75)*xq4;
    RB(76) = RB(76)*xq4;
    RF(77) = RF(77)*xq1;
    RB(80) = RB(80)*xq2;
    RB(81) = RB(81)*xq3;
    RB(82) = RB(82)*xq5;
    RB(85) = RB(85)*xq5;
    RF(87) = RF(87)*xq1;
    RB(87) = RB(87)*xq7;
    RF(88) = RF(88)*xq4;
    RF(89) = RF(89)*xq4;
    RB(90) = RB(90)*xq4;
    RF(91) = RF(91)*xq2;
    RF(92) = RF(92)*xq3;
    RB(94) = RB(94)*xq8;
    RF(96) = RF(96)*xq5;
    RF(97) = RF(97)*xq5;
    RF(98) = RF(98)*xq5;
    RB(98) = RB(98)*xq3;
    RF(99) = RF(99)*xq5;
    RF(100) = RF(100)*xq5;
    RF(101) = RF(101)*xq5;
    RF(105) = RF(105)*xq1;
    RF(106) = RF(106)*xq2;
    RF(107) = RF(107)*xq3;
    RB(108) = RB(108)*xq3;
    RF(111) = RF(111)*xq1;
    RF(112) = RF(112)*xq2;
    RB(112) = RB(112)*xq7;
    RB(114) = RB(114)*xq6;
    RF(115) = RF(115)*xq7;
    RB(117) = RB(117)*xq2;
    RF(120) = RF(120)*xq4;
    RB(120) = RB(120)*xq7;
    RF(122) = RF(122)*xq6;
    RF(123) = RF(123)*xq6;
    RB(123) = RB(123)*xq2;
    RF(124) = RF(124)*xq6;
    RF(125) = RF(125)*xq6;
    RB(125) = RB(125)*xq2;
    RB(126) = RB(126)*xq9;
    RB(130) = RB(130)*xq2;
    RF(132) = RF(132)*xq7;
    RF(133) = RF(133)*xq7;
    RF(134) = RF(134)*xq7;
    RB(134) = RB(134)*xq6;
    RF(135) = RF(135)*xq7;
    RF(136) = RF(136)*xq7;
    RF(137) = RF(137)*xq7;
    RF(138) = RF(138)*xq7;
    RF(139) = RF(139)*xq7;
    RB(139) = RB(139)*xq9;
    RF(140) = RF(140)*xq7;
    RB(140) = RB(140)*xq4;
    RF(141) = RF(141)*xq7;
    RB(141) = RB(141)*xq9;
    RF(142) = RF(142)*xq7;
    RF(144) = RF(144)*xq7;
    RF(145) = RF(145)*xq7;
    RF(146) = RF(146)*xq7;
    RF(147) = RF(147)*xq9;
    RF(148) = RF(148)*xq9;
    RF(149) = RF(149)*xq9;
    RB(149) = RB(149)*xq4;
    RF(150) = RF(150)*xq9;
    RF(151) = RF(151)*xq9;
    RF(152) = RF(152)*xq9;
    RF(153) = RF(153)*xq9;
    RF(154) = RF(154)*xq9;
    RB(155) = RB(155)*xq6;
    RB(156) = RB(156)*xq8;
    RB(157) = RB(157)*xq7;
    RB(158) = RB(158)*xq7;
    RB(159) = RB(159)*xq4;
    RB(160) = RB(160)*xq2;
    RB(161) = RB(161)*xq7;
    RB(162) = RB(162)*xq7;
    RF(164) = RF(164)*xq4;
    RB(164) = RB(164)*xq8;
    RF(165) = RF(165)*xq2;
    RF(166) = RF(166)*xq3;
    RB(166) = RB(166)*xq6;
    RF(167) = RF(167)*xq3;
    RB(168) = RB(168)*xq7;
    RB(169) = RB(169)*xq10;
    RF(170) = RF(170)*xq8;
    RF(171) = RF(171)*xq8;
    RF(172) = RF(172)*xq8;
    RF(173) = RF(173)*xq8;
    RF(174) = RF(174)*xq8;
    RF(175) = RF(175)*xq8;
    RF(176) = RF(176)*xq8;
    RF(177) = RF(177)*xq8;
    RF(178) = RF(178)*xq8;
    RB(180) = RB(180)*xq8;
    RB(181) = RB(181)*xq8;
    RB(182) = RB(182)*xq8;
    RF(183) = RF(183)*xq3;
    RB(183) = RB(183)*xq8;
    RB(184) = RB(184)*xq8;
    RB(186) = RB(186)*xq6;
    RB(188) = RB(188)*xq7;
    RF(189) = RF(189)*xq4;
    RB(190) = RB(190)*xq10;
    RF(199) = RF(199)*xq10;
    RB(199) = RB(199)*xq8;
    RF(200) = RF(200)*xq10;
    RF(201) = RF(201)*xq10;
    RB(201) = RB(201)*xq8;
    RF(202) = RF(202)*xq10;
    RF(203) = RF(203)*xq10;
    RF(204) = RF(204)*xq10;
    RB(204) = RB(204)*xq8;
    RF(205) = RF(205)*xq10;
}
template<class real> void cpu_rdwdot_kernel(const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt){
    real ROP5 = ROP2(5) + ROP2(6) + ROP2(7) + ROP2(8);
    real ROP12 = ROP2(12) + ROP2(13) + ROP2(14) + ROP2(15);

    WDOT(2) = (-ROP2(1) +ROP2(2) +ROP2(3) -ROP5 -ROP5
            -ROP2(9) -ROP2(10) -ROP12 -ROP2(17)
            -ROP2(18) -ROP2(19) -ROP2(24) -ROP2(25)
            +ROP2(30) +ROP2(34) +ROP2(35) +ROP2(36)
            +ROP2(37) -ROP2(41) -ROP2(42) +ROP2(44)
            +ROP2(46) -ROP2(48) +ROP2(49) +ROP2(50)
            +ROP2(52) +ROP2(52) +ROP2(53) +ROP2(57)
            -ROP2(60) +ROP2(62) +ROP2(63) +ROP2(64)
            +ROP2(65) -ROP2(71) -ROP2(72) +ROP2(77)
            -ROP2(78) +ROP2(79) +ROP2(87) +ROP2(91)
            +ROP2(92) +ROP2(94) -ROP2(96) -ROP2(97)
            -ROP2(98) -ROP2(102) +ROP2(105) -ROP2(108)
            +ROP2(109) +ROP2(115) +ROP2(116) +ROP2(118)
            +ROP2(124) -ROP2(126) -ROP2(127) -ROP2(128)
            -ROP2(132) -ROP2(133) -ROP2(134) +ROP2(135)
            +ROP2(146) -ROP2(148) -ROP2(149) -ROP2(150)
            -ROP2(156) -ROP2(157) +ROP2(165) +ROP2(167)
            -ROP2(170) -ROP2(171) +ROP2(173) -ROP2(180)
            -ROP2(185) -ROP2(186) -ROP2(190) -ROP2(191)
            -ROP2(192) +ROP2(193) -ROP2(199) -ROP2(200))*rateconv *molwt[1];
}
template<class real> void cpu_rdwdot2_kernel(const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt){

    WDOT(20) = (+ROP2(121) +ROP2(146) +ROP2(165) +ROP2(167)
            -ROP2(185) -ROP2(186) -ROP2(187) -ROP2(188)
            -ROP2(189) +ROP2(192) +ROP2(195) +ROP2(196)
            +ROP2(197) +ROP2(198) +ROP2(206))*rateconv *molwt[19];

    WDOT(21) = (ROP2(145) +ROP2(185) +ROP2(187) +ROP2(189)
            -ROP2(190) -ROP2(191) -ROP2(192) -ROP2(193)
            -ROP2(194) -ROP2(195) -ROP2(196) -ROP2(197)
            -ROP2(198) +ROP2(200) +ROP2(202) +ROP2(203)
            +ROP2(205))*rateconv *molwt[20];

    WDOT(22) = 0.0;
}
template<class real> void cpu_rdwdot3_kernel(const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt){

    real ROP114 =  ROP2(114) - ROP2(122);

    WDOT(13) = (+ROP2(31) +ROP2(37) +ROP2(41) +ROP2(53)
            +ROP2(55) +ROP2(63) +ROP2(70) -ROP2(71)
            -ROP2(72) -ROP2(73) -ROP2(74) -ROP2(75)
            -ROP2(76) -ROP2(77) +ROP2(79) +ROP2(83)
            -ROP2(90) +ROP2(96) +ROP2(99) +ROP2(100)
            +ROP2(101) +ROP2(140) +ROP2(154) +ROP2(160)
            +ROP2(172) +ROP2(177) +ROP2(188) +ROP2(201)
            +ROP2(204))*rateconv *molwt[12];
    WDOT(14) = (+ROP2(57) +ROP2(58) +ROP2(111) +ROP2(113)
            -ROP114 +ROP2(115) -ROP2(116) -ROP2(117)
            -ROP2(118) -ROP2(119) -ROP2(120) -ROP2(121)
            +ROP2(133) +ROP2(137) +ROP2(138) +ROP2(144))*rateconv *molwt[13];

    WDOT(15) = (+ROP2(91) +ROP2(92) +ROP2(95) +ROP2(105)
            +ROP2(132) +ROP2(142) +ROP2(143) -ROP2(155)
            -ROP2(156) -ROP2(157) -ROP2(158) -ROP2(159)
            -ROP2(160) -ROP2(161) -ROP2(162) -ROP2(163)
            -ROP2(164) -ROP2(165) -ROP2(166) -ROP2(167)
            -ROP2(168) -ROP2(169) +ROP2(171) +ROP2(174)
            +ROP2(176) +ROP2(191))*rateconv *molwt[14];
    WDOT(16) = (+ROP2(93) +ROP2(170) +ROP2(175) +ROP2(178)
            +ROP2(179) -ROP2(180) -ROP2(181) -ROP2(182)
            -ROP2(183) -ROP2(184))*rateconv *molwt[15];

    WDOT(17) = (+ROP2(39) -ROP2(95) -ROP2(108) -ROP2(109)
            -ROP2(110) -ROP2(111) -ROP2(112) -ROP2(113) -ROP2(113)
            +ROP2(116) +ROP2(127) +ROP2(129) +ROP2(131))*rateconv *molwt[16];
    WDOT(18) = (+ROP2(56) +ROP2(77) +ROP2(118) +ROP2(124)
            -ROP2(126) -ROP2(127) -ROP2(128) -ROP2(129)
            -ROP2(130) -ROP2(131) +ROP2(135) +ROP2(150)
            +ROP2(151) +ROP2(152) +ROP2(153) +ROP2(193))*rateconv *molwt[17];
    WDOT(19) = (+ROP2(89) +ROP2(148) +ROP2(163) +ROP2(173))*rateconv *molwt[18];
}
template<class real> void cpu_rdwdot6_kernel(const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt){

    WDOT(11) = (-ROP2(29) -ROP2(30) -ROP2(31) -ROP2(32)
            -ROP2(33) +ROP2(34) -ROP2(39) +ROP2(40)
            +ROP2(42) +ROP2(43) +ROP2(45) +ROP2(46)
            +ROP2(47) -ROP2(56) +ROP2(61) +ROP2(65)
            +ROP2(66) +ROP2(70) +ROP2(88) +ROP2(95)
            +ROP2(108) +ROP2(109) +ROP2(109) +ROP2(110) +ROP2(110)
            +ROP2(111) +ROP2(112) +ROP2(113) +ROP2(113)
            +ROP2(117) +ROP2(119) +ROP2(120) +ROP2(123)
            +ROP2(128) +ROP2(136) +ROP2(143) +ROP2(147)
            +ROP2(154) +ROP2(164) +ROP2(179) +ROP2(189))*rateconv *molwt[10];
    WDOT(12) = (+ROP2(29) +ROP2(30) +ROP2(32) +ROP2(33)
            -ROP2(40) +ROP2(44) +ROP2(52) -ROP2(70)
            +ROP2(125) +ROP2(130))*rateconv *molwt[11];
}
template<class real> void cpu_rdwdot7_kernel(const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt){

    WDOT(9) = (+ROP2(48) +ROP2(49) +ROP2(64) -ROP2(78)
            -ROP2(79) -ROP2(80) -ROP2(81) -ROP2(82)
            -ROP2(83) -ROP2(84) -ROP2(85) -ROP2(86)
            -ROP2(87) -ROP2(88) -ROP2(89) -ROP2(90)
            -ROP2(91) -ROP2(92) -ROP2(93) -ROP2(93)
            -ROP2(94) -ROP2(94) -ROP2(95) +ROP2(97)
            +ROP2(102) +ROP2(103) +ROP2(104) +ROP2(106) +ROP2(106)
            +ROP2(107) +ROP2(107) +ROP2(119) -ROP2(121)
            +ROP2(128) +ROP2(136) -ROP2(144) -ROP2(145)
            -ROP2(146) +ROP2(147) +ROP2(149) +ROP2(159)
            -ROP2(168) -ROP2(169) +ROP2(172) +ROP2(177)
            +ROP2(183) -ROP2(184) +ROP2(191) +ROP2(193)
            -ROP2(198) +ROP2(199) -ROP2(205) +ROP2(206))*rateconv *molwt[8];
    WDOT(10) = (+ROP2(78) +ROP2(84) +ROP2(86) +ROP2(88)
            +ROP2(90) -ROP2(102) -ROP2(103) -ROP2(104)
            -ROP2(105) -ROP2(106) -ROP2(107) +ROP2(144)
            +ROP2(166) +ROP2(168) +ROP2(184) +ROP2(186)
            +ROP2(198) +ROP2(205))*rateconv *molwt[9];

}
template<class real> void cpu_rdwdot8_kernel(const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt){
    real ROP12 = ROP2(12) + ROP2(13) + ROP2(14)+ ROP2(15);
    real ROP22 = ROP2(22) + ROP2(23);
    real ROP27 = ROP2(27) + ROP2(28);

    WDOT(7) = (+ROP12 -ROP2(17) -ROP2(18) -ROP2(19)
            -ROP2(20) -ROP2(21) -ROP22 -ROP22
            +ROP2(24) +ROP2(26) +ROP27 -ROP2(33)
            +ROP2(47) -ROP2(55) +ROP2(75) -ROP2(76)
            -ROP2(84) -ROP2(85) +ROP2(86) +ROP2(101)
            +ROP2(138) -ROP2(141) +ROP2(142) +ROP2(153)
            +ROP2(162) -ROP2(163) +ROP2(174) -ROP2(175)
            -ROP2(176) -ROP2(177) +ROP2(178) -ROP2(187)
            -ROP2(188) -ROP2(197) +ROP2(203)
            -ROP2(204))*rateconv *molwt[6];
    WDOT(8) = (+ROP2(16) +ROP22 -ROP2(24) -ROP2(25)
            -ROP2(26) -ROP27 +ROP2(76) -ROP2(86)
            -ROP2(142) +ROP2(176) -ROP2(178)
            +ROP2(197))*rateconv *molwt[7];
}
template<class real> void cpu_rdwdot9_kernel(const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt){
    real ROP27 = ROP2(27) + ROP2(28);
    WDOT(5) = (+ROP2(1) +ROP2(2) -ROP2(3) -ROP2(4) -ROP2(4)
            -ROP2(9) +ROP2(10) -ROP2(16) -ROP2(16)
            +ROP2(19) +ROP2(19) +ROP2(20) -ROP2(21)
            +ROP2(25) +ROP2(26) -ROP27 -ROP2(30)
            +ROP2(33) -ROP2(35) +ROP2(43) -ROP2(45)
            +ROP2(51) -ROP2(53) -ROP2(54) +ROP2(55)
            -ROP2(63) +ROP2(65) +ROP2(73) -ROP2(74)
            -ROP2(80) -ROP2(81) +ROP2(83) +ROP2(85)
            +ROP2(97) +ROP2(99) -ROP2(100) +ROP2(103)
            -ROP2(104) +ROP2(110) -ROP2(118) -ROP2(119)
            -ROP2(124) +ROP2(129) -ROP2(131) -ROP2(137)
            +ROP2(141) +ROP2(151) -ROP2(152) +ROP2(154)
            +ROP2(158) -ROP2(161) +ROP2(163) +ROP2(177)
            +ROP2(181) -ROP2(182) +ROP2(188) +ROP2(195)
            -ROP2(196) -ROP2(202) +ROP2(204))*rateconv *molwt[4];
}
template<class real> void cpu_rdwdot10_kernel(const real* RESTRICT RKF, const real* RESTRICT RKR,
        real* RESTRICT WDOT, real rateconv, const real* RESTRICT molwt){

    real ROP12 = ROP2(12) + ROP2(13) + ROP2(14)+ ROP2(15);
    real ROP22 = ROP2(22) + ROP2(23);
    real ROP27 = ROP2(27) + ROP2(28);
    real ROP5 = ROP2(5) + ROP2(6) + ROP2(7) + ROP2(8);

    WDOT(1) = (-ROP2(2) -ROP2(3) +ROP5 +ROP2(18)
            +ROP2(24) -ROP2(31) -ROP2(36) +ROP2(42)
            -ROP2(49) +ROP2(58) +ROP2(60) +ROP2(61)
            -ROP2(64) +ROP2(72) +ROP2(96) +ROP2(102)
            +ROP2(127) +ROP2(133) +ROP2(134) +ROP2(150)
            +ROP2(155) +ROP2(157) +ROP2(171) +ROP2(180)
            +ROP2(192) +ROP2(200))*rateconv*molwt[0] ;

    WDOT(3) = (+ROP2(1) -ROP2(2) +ROP2(4) -ROP2(10)
            -ROP2(11) -ROP2(11) +ROP2(17) -ROP2(20)
            -ROP2(26) -ROP2(29) +ROP2(32) -ROP2(34)
            +ROP2(38) -ROP2(43) -ROP2(44) -ROP2(50)
            -ROP2(61) -ROP2(62) -ROP2(73) -ROP2(79)
            +ROP2(82) -ROP2(99) -ROP2(103) -ROP2(109)
            -ROP2(116) -ROP2(117) -ROP2(123) -ROP2(129)
            -ROP2(130) -ROP2(135) -ROP2(136) +ROP2(139)
            -ROP2(151) -ROP2(158) -ROP2(159) -ROP2(160)
            -ROP2(172) -ROP2(173) -ROP2(181) -ROP2(193)
            -ROP2(194) -ROP2(195) -ROP2(201))*rateconv *molwt[2];

    WDOT(4) = (-ROP2(1) +ROP2(11) -ROP12 +ROP2(18)
            +ROP2(20) +ROP2(21) +ROP22 -ROP2(32)
            -ROP2(38) -ROP2(47) -ROP2(51) -ROP2(52)
            -ROP2(65) -ROP2(66) -ROP2(75) -ROP2(82)
            -ROP2(83) +ROP2(84) -ROP2(101) -ROP2(110)
            -ROP2(125) -ROP2(138) -ROP2(139) -ROP2(140)
            -ROP2(153) -ROP2(154) -ROP2(162) -ROP2(174)
            +ROP2(175) +ROP2(187) -ROP2(203))*rateconv *molwt[3];
    WDOT(6) = (+ROP2(3) +ROP2(4) +ROP2(9) +ROP2(17)
            +ROP2(21) +ROP2(25) +ROP27 -ROP2(37)
            +ROP2(45) +ROP2(54) +ROP2(66) +ROP2(74)
            +ROP2(80) +ROP2(81) +ROP2(98) +ROP2(100)
            +ROP2(104) +ROP2(131) +ROP2(137) +ROP2(152)
            +ROP2(161) +ROP2(182) +ROP2(196) +ROP2(202))*rateconv *molwt[5];
}
template<class real>void run(const char*name){real gpu_t[1]={real(1000)},gpu_p[1]={real(1.0132e6)},gpu_y[22]={},gpu_molwt[22],gpu_wdot[22]={},gpu_rf[206]={},gpu_rb[206]={},gpu_rklow[21]={},gpu_c[22]={},gpu_a[121]={},gpu_eg[32]={};gpu_y[14]=real(.064);gpu_y[3]=real(.218);gpu_y[21]=real(.718);for(auto&x:gpu_molwt)x=1;real tconv=1,pconv=1,rateconv=1;
cpu_ratt_kernel(gpu_t,gpu_rf,tconv);
cpu_rdsmh_kernel(gpu_t,gpu_eg,tconv);
cpu_gr_base(gpu_p,gpu_t,gpu_y,gpu_c,tconv,pconv);
cpu_ratt2_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
cpu_ratt3_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
cpu_ratt4_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
cpu_ratt5_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
cpu_ratt6_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
cpu_ratt7_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
cpu_ratt8_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
cpu_ratt9_kernel(gpu_t,gpu_rf,gpu_rb,gpu_eg,tconv);
cpu_ratt10_kernel(gpu_t,gpu_rklow,tconv);
cpu_ratx_kernel(gpu_t,gpu_c,gpu_rf,gpu_rb,gpu_rklow,tconv);
cpu_ratxb_kernel(gpu_t,gpu_c,gpu_rf,gpu_rb,gpu_rklow,tconv);
cpu_ratx2_kernel(gpu_c,gpu_rf,gpu_rb);
cpu_ratx4_kernel(gpu_c,gpu_rf,gpu_rb);
cpu_qssa_kernel(gpu_rf,gpu_rb,gpu_a);
cpu_qssab_kernel(gpu_rf,gpu_rb,gpu_a);
cpu_qssa2_kernel(gpu_rf,gpu_rb,gpu_a);
cpu_rdwdot_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
cpu_rdwdot2_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
cpu_rdwdot3_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
cpu_rdwdot6_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
cpu_rdwdot7_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
cpu_rdwdot8_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
cpu_rdwdot9_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
cpu_rdwdot10_kernel(gpu_rf,gpu_rb,gpu_wdot,rateconv,gpu_molwt);
std::ofstream out(name,std::ios::binary);for(auto spec: {std::pair<real*,int>{gpu_wdot,22},{gpu_c,22},{gpu_rf,206},{gpu_rb,206},{gpu_rklow,21},{gpu_eg,31}})for(int j=0;j<spec.second;j++){double x=spec.first[j];if(!std::isfinite(x)){fprintf(stderr,"NonfiniteCPUstate\n");exit(1);}out.write((char*)&x,8);}for(int i=0;i<22;i++)printf("%d %.17g\n",i,(double)gpu_wdot[i]);}
int main(){run<float>("study_cpu_f32_f64.bin");run<double>("study_cpu_f64_f64.bin");}
