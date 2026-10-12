#define render_kernel aobench_native_render_kernel
#define render aobench_native_render
#define main aobench_native_main
#include "../aobench-cuda/ao.cu"
#undef main
#undef render
#undef render_kernel

extern "C" __global__ void render_kernel(unsigned char* fimg, const Sphere* spheres,
                                          const Plane plane, const int h, const int w,
                                          const int nsubsamples)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (y < h && x < w) {
    RNG rng(y * w + x);
    float s0 = 0;
    float s1 = 0;
    float s2 = 0;

    for (int v = 0; v < nsubsamples; v++) {
      for (int u = 0; u < nsubsamples; u++) {
        float px = (x + (u / (float)nsubsamples) - (w / 2.0f)) / (w / 2.0f);
        float py = -(y + (v / (float)nsubsamples) - (h / 2.0f)) / (h / 2.0f);

        Ray ray;
        ray.org.x = 0.f;
        ray.org.y = 0.f;
        ray.org.z = 0.f;
        ray.dir.x = px;
        ray.dir.y = py;
        ray.dir.z = -1.f;
        vnormalize(&(ray.dir));

        Isect isect;
        isect.t = 1.0e+17f;
        isect.hit = 0;

        ray_sphere_intersect(&isect, &ray, spheres);
        ray_sphere_intersect(&isect, &ray, spheres + 1);
        ray_sphere_intersect(&isect, &ray, spheres + 2);
        ray_plane_intersect(&isect, &ray, &plane);

        if (isect.hit) {
          vec col;
          ambient_occlusion(&col, &isect, spheres, &plane, rng);
          s0 += col.x;
          s1 += col.y;
          s2 += col.z;
        }
      }
    }
    fimg[3 * (y * w + x) + 0] = clamp(s0 / (float)(nsubsamples * nsubsamples));
    fimg[3 * (y * w + x) + 1] = clamp(s1 / (float)(nsubsamples * nsubsamples));
    fimg[3 * (y * w + x) + 2] = clamp(s2 / (float)(nsubsamples * nsubsamples));
  }
}
