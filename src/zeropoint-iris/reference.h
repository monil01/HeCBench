#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

inline void reference(
    const float* x_min,
    const float* x_max,
    int32_t qmin,
    int32_t qmax,
    int size,
    bool preserve_sparsity,
    float* scale,
    int32_t* zero_point)
{
  for (int i = 0; i < size; i++) {
    float min_val = x_min[i];
    float max_val = x_max[i];

    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      int symmetric_qmin = -((qmax - qmin) / 2 + 1);
      int symmetric_qmax = (qmax - qmin) / 2;
      double max_scale = std::max(
          std::fabs(min_val / symmetric_qmin),
          std::fabs(max_val / symmetric_qmax));
      min_val = max_scale * symmetric_qmin;
      max_val = max_scale * symmetric_qmax;
    }

    min_val = std::min(min_val, 0.f);
    max_val = std::max(max_val, 0.f);
    scale[i] = (static_cast<double>(max_val) - min_val) / (qmax - qmin);

    if (scale[i] == 0.0f || std::isinf(1.0f / scale[i])) {
      scale[i] = 0.1f;
    }

    double zero_point_from_min = qmin - min_val / static_cast<double>(scale[i]);
    double zero_point_from_max = qmax - max_val / static_cast<double>(scale[i]);
    double zero_point_from_min_error =
        std::abs(qmin) + std::abs(min_val / static_cast<double>(scale[i]));
    double zero_point_from_max_error =
        std::abs(qmax) + std::abs(max_val / static_cast<double>(scale[i]));
    double initial_zero_point =
        zero_point_from_min_error < zero_point_from_max_error
            ? zero_point_from_min
            : zero_point_from_max;

    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      initial_zero_point = static_cast<double>(qmin + qmax) / 2;
    }

    int32_t nudged_zero_point = 0;
    if (initial_zero_point < qmin) {
      nudged_zero_point = qmin;
    } else if (initial_zero_point > qmax) {
      nudged_zero_point = qmax;
    } else {
      nudged_zero_point = std::nearbyint(initial_zero_point);
    }
    zero_point[i] = nudged_zero_point;
  }
}
