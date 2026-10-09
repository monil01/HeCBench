#ifndef PROJECTILE_IRIS_REFERENCE_HPP
#define PROJECTILE_IRIS_REFERENCE_HPP

#include "Projectile.hpp"

constexpr float kPIValue = 3.1415f;
constexpr float kGValue = 9.81f;

void reference(const Projectile* obj, Projectile* pObj, const int num_elements) {
  for (int i = 0; i < num_elements; i++) {
    float proj_angle = obj[i].getangle();
    float proj_vel = obj[i].getvelocity();
    float sin_value = std::sin(proj_angle * kPIValue / 180.0f);
    float cos_value = std::cos(proj_angle * kPIValue / 180.0f);
    float total_time = std::fabs((2 * proj_vel * sin_value)) / kGValue;
    float max_range = std::fabs(proj_vel * total_time * cos_value);
    float max_height =
        (proj_vel * proj_vel * sin_value * sin_value) / 2.0f * kGValue;
    pObj[i].setRangeandTime(max_range, total_time, proj_angle, proj_vel,
                            max_height);
  }
}

#endif
