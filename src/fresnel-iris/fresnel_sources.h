#ifndef FRESNEL_SOURCES_H
#define FRESNEL_SOURCES_H

#include "xchebyshev.cu"

#define sqrt_2pi cosine_sqrt_2pi
#define Chebyshev_Expansion_0_1 cosine_Chebyshev_Expansion_0_1
#define Chebyshev_Expansion_1_3 cosine_Chebyshev_Expansion_1_3
#define Chebyshev_Expansion_3_5 cosine_Chebyshev_Expansion_3_5
#define Chebyshev_Expansion_5_7 cosine_Chebyshev_Expansion_5_7
#define Asymptotic_Series cosine_Asymptotic_Series
#include "cosine.cu"
#undef sqrt_2pi
#undef Chebyshev_Expansion_0_1
#undef Chebyshev_Expansion_1_3
#undef Chebyshev_Expansion_3_5
#undef Chebyshev_Expansion_5_7
#undef Asymptotic_Series

#define sqrt_2pi sine_sqrt_2pi
#define Chebyshev_Expansion_0_1 sine_Chebyshev_Expansion_0_1
#define Chebyshev_Expansion_1_3 sine_Chebyshev_Expansion_1_3
#define Chebyshev_Expansion_3_5 sine_Chebyshev_Expansion_3_5
#define Chebyshev_Expansion_5_7 sine_Chebyshev_Expansion_5_7
#define Asymptotic_Series sine_Asymptotic_Series
#include "sine.cu"
#undef sqrt_2pi
#undef Chebyshev_Expansion_0_1
#undef Chebyshev_Expansion_1_3
#undef Chebyshev_Expansion_3_5
#undef Chebyshev_Expansion_5_7
#undef Asymptotic_Series

#include "fresnel.cu"

#endif
