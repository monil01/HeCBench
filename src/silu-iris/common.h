#pragma once

#include <cstddef>
#include <cstdlib>

template<class T, class P>
T ceil_div(T dividend, P divisor) {
    return (dividend + divisor-1) / divisor;
}

inline int max_int(int a, int b) {
    return a > b ? a : b;
}

float* make_random_float(size_t N) {
    float* arr = (float*)malloc(N * sizeof(float));
    #pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
        arr[i] = rand() / (float)RAND_MAX * 2.f - 1.f;
    }
    return arr;
}
