#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <hip/hip_runtime.h>
static void diagnostic_dump(const char* name, const void* device, int width, int height, int depth, size_t pitch) {
  const char* prefix = getenv("HECBENCH_DUMP");
  if (!prefix) return;
  std::vector<float> out(size_t(width)*height*depth);
  for (int z=0;z<depth;++z) {
    auto err = hipMemcpy2D(out.data()+size_t(z)*height*width, size_t(width)*sizeof(float),
        static_cast<const char*>(device)+size_t(z)*height*pitch, pitch,
        size_t(width)*sizeof(float), height, hipMemcpyDeviceToHost);
    if (err != hipSuccess) exit(2);
  }
  std::string path=std::string(prefix)+"."+name+"."+std::to_string(width)+"x"+std::to_string(height)+"x"+std::to_string(depth)+".raw";
  FILE* f=fopen(path.c_str(),"wb");
  if(!f || fwrite(out.data(),sizeof(float),out.size(),f)!=out.size() || fclose(f)) exit(3);
}
