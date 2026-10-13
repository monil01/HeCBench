#pragma once
inline void dispatch_PetrinetKernel(dim3 grid, dim3 block, int* __restrict__ g_s, float* __restrict__ g_v, int*__restrict__ g_m, int n, int s, int seed) { iris_launch("PetrinetKernel", grid, block, g_s, g_v, g_m, n, s, seed); }
