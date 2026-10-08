typedef unsigned char uchar;

extern "C" __global__ void base64_enc(const uchar* __restrict__ input,
                                       uchar* __restrict__ output,
                                       const char padCount,
                                       const unsigned long numBlock,
                                       const unsigned long offset) {
  const uchar base64chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  const size_t id = blockIdx.x * blockDim.x + threadIdx.x + offset;
  if (id >= numBlock) return;

  const bool last = (padCount != 0) & (id == numBlock - 1);
  const uchar3 loaded = reinterpret_cast<const uchar3*>(input)[id];
  const uchar si = loaded.x;
  const uchar si1 = loaded.y;
  const uchar si2 = loaded.z;

  uchar4 result;
  result.x = base64chars[si / 4];
  result.y = (last && padCount == 1) ? base64chars[(si * 16) % 64]
                                     : base64chars[(si * 16) % 64 + si1 / 16];
  result.z = last ? ((padCount == 1) ? '=' : base64chars[(si1 * 4) % 64])
                  : base64chars[(si1 * 4) % 64 + si2 / 64];
  result.w = last ? '=' : base64chars[si2 % 64];
  reinterpret_cast<uchar4*>(output)[id] = result;
}
