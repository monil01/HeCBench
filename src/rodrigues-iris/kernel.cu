struct Float3 {
  float x;
  float y;
  float z;
};

struct Float4 {
  float x;
  float y;
  float z;
  float w;
};

extern "C" __global__ void rotate(const int n, const float angle, const Float3 w, Float3* d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  float s;
  float c;
  sincosf(angle, &s, &c);

  const Float3 p = d[i];
  const float mc = 1.f - c;

  const float m1 = c + (w.x) * (w.x) * (mc);
  const float m2 = (w.z) * s + (w.x) * (w.y) * (mc);
  const float m3 = -(w.y) * s + (w.x) * (w.z) * (mc);

  const float m4 = -(w.z) * s + (w.x) * (w.y) * (mc);
  const float m5 = c + (w.y) * (w.y) * (mc);
  const float m6 = (w.x) * s + (w.y) * (w.z) * (mc);

  const float m7 = (w.y) * s + (w.x) * (w.z) * (mc);
  const float m8 = -(w.x) * s + (w.y) * (w.z) * (mc);
  const float m9 = c + (w.z) * (w.z) * (mc);

  const float ox = p.x * m1 + p.y * m2 + p.z * m3;
  const float oy = p.x * m4 + p.y * m5 + p.z * m6;
  const float oz = p.x * m7 + p.y * m8 + p.z * m9;
  d[i] = { ox, oy, oz };
}

extern "C" __global__ void rotate2(const int n, const float angle, const Float3 w, Float4* d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  float s;
  float c;
  sincosf(angle, &s, &c);

  const Float4 p = d[i];
  const float mc = 1.f - c;

  const float m1 = c + (w.x) * (w.x) * (mc);
  const float m2 = (w.z) * s + (w.x) * (w.y) * (mc);
  const float m3 = -(w.y) * s + (w.x) * (w.z) * (mc);

  const float m4 = -(w.z) * s + (w.x) * (w.y) * (mc);
  const float m5 = c + (w.y) * (w.y) * (mc);
  const float m6 = (w.x) * s + (w.y) * (w.z) * (mc);

  const float m7 = (w.y) * s + (w.x) * (w.z) * (mc);
  const float m8 = -(w.x) * s + (w.y) * (w.z) * (mc);
  const float m9 = c + (w.z) * (w.z) * (mc);

  const float ox = p.x * m1 + p.y * m2 + p.z * m3;
  const float oy = p.x * m4 + p.y * m5 + p.z * m6;
  const float oz = p.x * m7 + p.y * m8 + p.z * m9;
  d[i] = { ox, oy, oz, 0.f };
}
