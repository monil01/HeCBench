#pragma once

#define COL_P_X 0
#define COL_P_Y 1
#define COL_P_Z 2
#define COL_N_X 3
#define COL_N_Y 4
#define COL_N_Z 5
#define COL_RSq 6
#define COL_DIM 7

template <typename T>
void reference(const T* s, int n, T f, int w, int h, T* d)
{
#pragma omp parallel for collapse(2)
  for (int idy = 0; idy < h; idy++) {
    for (int idx = 0; idx < w; idx++) {
      T ray[3];
      ray[0] = T(idx) - (w - 1) * T(0.5);
      ray[1] = T(idy) - (h - 1) * T(0.5);
      ray[2] = f;
      T dMin = 1e20;

      for (int i = 0; i < n; ++i) {
        T p[3];
        T normal[3];
        p[0] = s[i * COL_DIM + COL_P_X];
        p[1] = s[i * COL_DIM + COL_P_Y];
        p[2] = s[i * COL_DIM + COL_P_Z];
        normal[0] = s[i * COL_DIM + COL_N_X];
        normal[1] = s[i * COL_DIM + COL_N_Y];
        normal[2] = s[i * COL_DIM + COL_N_Z];
        T rSqMax = s[i * COL_DIM + COL_RSq];
        T pDotn = p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2];
        T dsDotRay = ray[0] * normal[0] + ray[1] * normal[1] + ray[2] * normal[2];
        T alpha = pDotn / dsDotRay;
        T pt0 = ray[0] * alpha - p[0];
        T pt1 = ray[1] * alpha - p[1];
        T pt2 = ray[2] * alpha - p[2];
        T t = ray[2] * alpha;
        T rSq = pt0 * pt0 + pt1 * pt1 + pt2 * pt2;
        if (rSq < rSqMax && dMin > t) dMin = t;
      }
      d[idy * w + idx] = dMin > T(100) ? T(0) : dMin;
    }
  }
}
