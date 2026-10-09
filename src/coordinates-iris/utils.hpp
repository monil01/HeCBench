#pragma once

#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

constexpr double DEGREE_TO_RADIAN = M_PI / 180.0;
constexpr double EARTH_CIRCUMFERENCE_EQUATOR_KM = 40000.0;
constexpr double EARTH_CIRCUMFERENCE_KM_PER_DEGREE =
    EARTH_CIRCUMFERENCE_EQUATOR_KM / 360.0;

template <typename T>
struct alignas(2 * sizeof(T)) vec_2d {
  using value_type = T;
  value_type x;
  value_type y;
};

template <typename T>
struct alignas(2 * sizeof(T)) lonlat_2d : vec_2d<T> {};

template <typename T>
struct alignas(2 * sizeof(T)) cartesian_2d : vec_2d<T> {};

template <typename T>
inline bool operator==(vec_2d<T> const& lhs, vec_2d<T> const& rhs)
{
  return std::fabs(lhs.x - rhs.x) < 1e-3 && std::fabs(lhs.y - rhs.y) < 1e-3;
}

template <typename T>
HOST_DEVICE inline T midpoint(T a, T b)
{
  return (a + b) / static_cast<T>(2);
}

template <typename T>
HOST_DEVICE inline T lon_to_x(T lon, T lat)
{
  return lon * EARTH_CIRCUMFERENCE_KM_PER_DEGREE * cos(lat * DEGREE_TO_RADIAN);
}

template <typename T>
HOST_DEVICE inline T lat_to_y(T lat)
{
  return lat * EARTH_CIRCUMFERENCE_KM_PER_DEGREE;
}

template <typename T>
struct to_cartesian_functor {
  HOST_DEVICE explicit to_cartesian_functor(lonlat_2d<T> origin) : origin_(origin) {}

  HOST_DEVICE cartesian_2d<T> operator()(lonlat_2d<T> loc) const
  {
    cartesian_2d<T> t;
    t.x = lon_to_x(origin_.x - loc.x, midpoint(loc.y, origin_.y));
    t.y = lat_to_y(origin_.y - loc.y);
    return t;
  }

  lonlat_2d<T> origin_{};
};
