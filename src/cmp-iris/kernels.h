extern "C" __global__ void
init_c(real *c, real inc, real c0) 
{
  int i = blockIdx.x;
  c[i] = c0 + inc*i;
}

extern "C" __global__ void
init_half(const real* __restrict__ scalco, 
          const real* __restrict__ gx, 
          const real* __restrict__ gy, 
          const real* __restrict__ sx, 
          const real* __restrict__ sy, 
          real* __restrict__ h) 
{
  int i = blockIdx.x;
  real _s = scalco[i];

  if(-EPSILON < _s && _s < EPSILON) _s = 1.0f;
  else if(_s < 0) _s = 1.0f / _s;

  real hx = (gx[i] - sx[i]) * _s;
  real hy = (gy[i] - sy[i]) * _s;

  h[i] = 0.25f * (hx * hx + hy * hy) / FACTOR;
}

extern "C" __global__ void
compute_semblances(const real* __restrict__ h, 
                   const real* __restrict__ c, 
                   const real* __restrict__ samples, 
                   real* __restrict__ num,
                   real* __restrict__ stt,
                   int t_id0, 
                   int t_idf,
                   real _idt,
                   real _dt,
                   int _tau,
                   int _w,
                   int nc,
                   int ns) 
{
  real _den = 0.0f, _ac_linear = 0.0f, _ac_squared = 0.0f;
  real _num[MAX_W],  m = 0.0f;
  int err = 0;

  int i = blockIdx.x * NTHREADS + threadIdx.x;

  int t0 = i / nc;
  int c_id = i % nc;

  if(i < ns * nc)
  {
    real _c = c[c_id];
    real _t0 = _dt * t0;
    _t0 *= _t0;

    for(int j=0; j < _w; j++) _num[j] = 0.0f;

    for(int t_id=t_id0; t_id < t_idf; t_id++) {
      real t = sqrtf(_t0 + _c * h[t_id]) * _idt;

      int it = (int)( t );
      int ittau = it - _tau;
      real x = t - (real)it;

      if(ittau >= 0 && it + _tau + 1 < ns) {
        int k1 = ittau + (t_id-t_id0)*ns;
        real sk1p1 = samples[k1], sk1;

        for(int j=0; j < _w; j++) {
          k1++;
          sk1 = sk1p1;
          sk1p1 = samples[k1];
          // linear interpolation optmized for this problema
          real v = (sk1p1 - sk1) * x + sk1;

          _num[j] += v;
          _den += v * v;
          _ac_linear += v;
        }
        m += 1;
      } else { err++; }
    }

    // Reduction for num
    for(int j=0; j < _w; j++) _ac_squared += _num[j] * _num[j];

    // Evaluate semblances
    if(_den > EPSILON && m > EPSILON && _w > EPSILON && err < 2) {
      num[i] = _ac_squared / (_den * m);
      stt[i] = _ac_linear  / (_w   * m);
    }
    else {
      num[i] = -1.0f;
      stt[i] = -1.0f;
    }
  }
}

extern "C" __global__ void
redux_semblances(const real* __restrict__ num, 
                 const real* __restrict__ stt, 
                 int*  __restrict__ ctr, 
                 real* __restrict__ str, 
                 real* __restrict__ stk,
                 const int nc, 
                 const int cdp_id,
                 const int ns) 
{
  int t0 = blockIdx.x * NTHREADS + threadIdx.x;

  if(t0 < ns)
  {
    real max_sem = 0.0f;
    int max_c = -1;

    for(int it=t0*nc; it < (t0+1)*nc ; it++) {
      real _num = num[it];
      if(_num > max_sem) {
        max_sem = _num;
        max_c = it;
      }
    }

    ctr[cdp_id*ns + t0] = max_c % nc;
    str[cdp_id*ns + t0] = max_sem;
    stk[cdp_id*ns + t0] = max_c > -1 ? stt[max_c] : 0;
  }
}
