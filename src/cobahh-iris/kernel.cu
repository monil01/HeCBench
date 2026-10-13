#include <cuda_runtime.h>

__device__  __host__
inline int _timestep(float t, float dt)
{
  return (int)((t + 1e-3f*dt)/dt); 
}

extern "C" __global__ void cobahh (
    float* __restrict__ d_h, 
    float* __restrict__ d_m,
    float* __restrict__ d_n,
    float* __restrict__ d_ge,
    float* __restrict__ d_v,
    float* __restrict__ d_gi,
    const float* __restrict__ d_lastspike, 
    char* __restrict__ d_not_refractory, 
    const int _N ,
    const float dt,
    const float t,
    const int    _lio_1,
    const float  _lio_2,
    const float  _lio_3,
    const float  _lio_4,
    const float  _lio_5,
    const float  _lio_6,
    const float  _lio_7,
    const float  _lio_8,
    const float  _lio_9,
    const float _lio_10,
    const float _lio_11,
    const float _lio_12,
    const float _lio_13,
    const float _lio_14,
    const float _lio_15,
    const float _lio_16,
    const float _lio_17,
    const float _lio_18,
    const float _lio_19,
    const float _lio_20,
    const float _lio_21,
    const float _lio_22,
    const float _lio_23,
    const float _lio_24,
    const float _lio_25,
    const float _lio_26,
    const float _lio_27,
    const float _lio_28,
    const float _lio_29,
    const float _lio_30,
    const float _lio_31,
    const float _lio_32,
    const float _lio_33
    )

{
  int _idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (_idx >= _N) return;
  float h = d_h[_idx];
  float m = d_m[_idx];
  float n = d_n[_idx];
  float ge = d_ge[_idx];
  float v = d_v[_idx];
  const float lastspike = d_lastspike[_idx];
  float gi = d_gi[_idx];
  char not_refractory;
  not_refractory = _timestep(t - lastspike, dt) >= _lio_1;
  const float _BA_h = (_lio_2 * expf(_lio_3 * v))/(((-4.0f)/(0.001f + (_lio_4 * expf(_lio_5 * v)))) - (_lio_2 * expf(_lio_3 * v)));
  const float _h = (- _BA_h) + ((_BA_h + h) * expf(dt * (((-4.0f)/(0.001f + (_lio_4 * expf(_lio_5 * v)))) - (_lio_2 * expf(_lio_3 * v)))));
  const float _BA_m = (((_lio_6/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + (_lio_10/(_lio_7 + (_lio_8 * expf(_lio_9 * v))))) - ((0.32f * v)/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))))/(((((_lio_11/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + (_lio_12/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))) + (_lio_16/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))) + ((0.32f * v)/(_lio_7 + (_lio_8 * expf(_lio_9 * v))))) - ((_lio_10/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + ((0.28f * v)/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))));
  const float _m = (- _BA_m) + ((_BA_m + m) * expf(dt * (((((_lio_11/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + (_lio_12/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))) + (_lio_16/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))) + ((0.32f * v)/(_lio_7 + (_lio_8 * expf(_lio_9 * v))))) - ((_lio_10/(_lio_7 + (_lio_8 * expf(_lio_9 * v)))) + ((0.28f * v)/(_lio_13 + (_lio_14 * expf(_lio_15 * v))))))));
  const float _BA_n = (((_lio_17/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + (_lio_19/(_lio_7 + (_lio_18 * expf(_lio_5 * v))))) - ((0.032f * v)/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))))/(((_lio_20/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + ((0.032f * v)/(_lio_7 + (_lio_18 * expf(_lio_5 * v))))) - ((_lio_19/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + (_lio_21 * expf(_lio_22 * v))));
  const float _n = (- _BA_n) + ((_BA_n + n) * expf(dt * (((_lio_20/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + ((0.032f * v)/(_lio_7 + (_lio_18 * expf(_lio_5 * v))))) - ((_lio_19/(_lio_7 + (_lio_18 * expf(_lio_5 * v)))) + (_lio_21 * expf(_lio_22 * v))))));
  const float _ge = _lio_23 * ge;
  const float _BA_v = (_lio_24 + ((((_lio_25 * (n*n*n*n)) + (_lio_26 * (h * (m*m*m)))) + (_lio_27 * ge)) + (_lio_28 * gi)))/((_lio_29 + (_lio_30 * (n*n*n*n))) - (((_lio_31 * (h * (m*m*m))) + (_lio_32 * ge)) + (_lio_32 * gi)));
  const float _v = (- _BA_v) + ((_BA_v + v) * expf(dt * ((_lio_29 + (_lio_30 * (n*n*n*n))) - (((_lio_31 * (h * (m*m*m))) + (_lio_32 * ge)) + (_lio_32 * gi)))));
  const float _gi = _lio_33 * gi;

  d_h[_idx] = _h;
  d_m[_idx] = _m;
  d_n[_idx] = _n;
  d_ge[_idx] = _ge;
  d_v[_idx] = _v;
  d_gi[_idx] = _gi;
  d_not_refractory[_idx] = not_refractory;
}
