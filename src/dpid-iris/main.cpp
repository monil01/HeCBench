// Copyright (c) 2016 Nicolas Weber and Sandra C. Amend / GCC / TU-Darmstadt.
// All rights reserved.
// Use of this source code is governed by the BSD 3-Clause license that can be
// found in the LICENSE file.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <iris/iris.hpp>
#include <chrono>
#include <vector>
#include <stdexcept>
#include "shared.h"
static void check(int rc) {if(rc!=IRIS_SUCCESS) throw std::runtime_error("IRIS API failure");}
#include "reference.h"

double LCG_random_double(uint64_t * seed)
{
  const uint64_t m = 9223372036854775808ULL; // 2^63
  const uint64_t a = 2806196910506780709ULL;
  const uint64_t c = 1ULL;
  *seed = (a * (*seed) + c) % m;
  return (double) (*seed) / (double) m;
}

void run(const Params& p, const void* hInput, void* hOutput) {
  size_t input_bytes=size_t(p.iWidth)*p.iHeight*3;
  size_t output_bytes=size_t(p.oWidth)*p.oHeight*3;
  std::vector<unsigned char> guidance(output_bytes);
  iris::DMem input(const_cast<void*>(hInput),input_bytes), output(hOutput,output_bytes), patches(guidance.data(),output_bytes);
  int count=0,device=-1; check(iris_device_count(&count));
  for(int d=0;d<count;++d) { int backend=0,type=0; char name[256]={};
    check(iris_device_info(d,iris_backend,&backend,nullptr)); check(iris_device_info(d,iris_type,&type,nullptr));
    check(iris_device_info(d,iris_name,name,nullptr));
    printf("IRIS device %d: %s backend=%d type=%d\n",d,name,backend,type);
    if(backend==iris_hip && (type&iris_gpu)) {device=d;break;}
  }
  if(device<0) throw std::runtime_error("no IRIS HIP GPU");
  size_t gws[]={size_t((p.oWidth+3)/4)*128,size_t(p.oHeight)}, lws[]={128,1};
  Params params=p;
  auto start=std::chrono::steady_clock::now();
  for(uint32_t i=0;i<p.repeat;++i) {
    iris::Task guide;
    void* ga[]={&input,&patches,&params}; int gi[]={iris_r,iris_w,sizeof(params)};
    check(guide.kernel("kernelGuidance",2,nullptr,gws,lws,3,ga,gi));
    check(guide.submit(device,nullptr,true));
    iris::Task down;
    void* da[]={&input,&patches,&params,&output}; int di[]={iris_r,iris_r,sizeof(params),iris_w};
    check(down.kernel("kernelDownsampling",2,nullptr,gws,lws,4,da,di));
    check(down.submit(device,nullptr,true));
  }
  printf("Average kernel execution time %f (s)\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/p.repeat);
  iris::Task flush; check(flush.flush_out(output)); check(flush.submit(device,nullptr,true));
}


int main(int argc, char** argv) {
  // generate a random image for testing
  if(argc != 5) {
    printf("Usage: %s <output image width> <output image height> ", argv[0]);
    // explore the influence of lambda
    printf("<lambda> <repeat>\n");
    exit(1);
  }

  // read params
  Params p;
  p.oWidth = (uint32_t)std::atoi(argv[1]);
  p.oHeight = (uint32_t)std::atoi(argv[2]);
  p.lambda = (float)std::atof(argv[3]);
  p.repeat = (uint32_t)std::atoi(argv[4]);

  // check params
  if(p.oWidth == 0 && p.oHeight == 0) {
    printf("only one dimension (width or height) can be 0!\n");
    exit(1);
  }

  p.iWidth  = 8192;
  p.iHeight = 8192;

  // calc width/height according to aspect ratio
  if(p.oWidth == 0)
    p.oWidth = (uint32_t)std::round((p.oHeight / (double)p.iHeight) * p.iWidth);

  if(p.oHeight == 0)
    p.oHeight = (uint32_t)std::round((p.oWidth / (double)p.iWidth) * p.iHeight);

  // calc patch size
  p.pWidth  = p.iWidth  / (float)p.oWidth;
  p.pHeight  = p.iHeight / (float)p.oHeight;

  // set random values for an image
  uchar3 *hInput   = (uchar3*) malloc (sizeof(uchar3) * p.iWidth * p.iHeight);
  uchar3 *hOutput  = (uchar3*) malloc (sizeof(uchar3) * p.oWidth * p.oHeight);

  uint64_t seed = 123;
  for (uint32_t i = 0; i < p.iWidth * p.iHeight; i++) {
    hInput[i].x = (unsigned char)(256*LCG_random_double(&seed));
    hInput[i].y = (unsigned char)(256*LCG_random_double(&seed));
    hInput[i].z = (unsigned char)(256*LCG_random_double(&seed));
  }

  iris::Platform platform; check(platform.init(&argc,&argv,true));
  // run downsampling on a device
  run(p, hInput, hOutput);

  // per-pixel comparison for validation
  uchar3 *hRef = (uchar3*) malloc(sizeof(uchar3) * p.oWidth * p.oHeight);
  reference(p.oWidth, p.oHeight, p.iWidth, p.iHeight,
            p.pWidth, p.pHeight, p.lambda,
            hInput, hRef);
  uint32_t mismatches = 0;
  int maxDiff = 0;
  for (uint32_t i = 0; i < p.oWidth * p.oHeight; i++) {
    int dx = std::abs((int)hOutput[i].x - (int)hRef[i].x);
    int dy = std::abs((int)hOutput[i].y - (int)hRef[i].y);
    int dz = std::abs((int)hOutput[i].z - (int)hRef[i].z);
    if (dx > 0 || dy > 0 || dz > 0) mismatches++;
    if (dx > maxDiff) maxDiff = dx;
    if (dy > maxDiff) maxDiff = dy;
    if (dz > maxDiff) maxDiff = dz;
  }
  uint32_t total = p.oWidth * p.oHeight;
  if (mismatches == 0) {
    printf("Verification PASS: GPU matches CPU reference exactly.\n");
  } else {
    printf("Verification: %u / %u pixels differ (%.2f%%), max channel diff = %d\n",
           mismatches, total, 100.0 * mismatches / total, maxDiff);
    if (maxDiff <= 1)
      printf("  -> within rounding tolerance (maxDiff<=1, likely OK)\n");
    else
      printf("  -> WARNING: differences exceed rounding tolerance\n");
  }

  uint64_t hash=14695981039346656037ULL;
  for(size_t j=0;j<size_t(p.oWidth)*p.oHeight*3;++j) {hash^=reinterpret_cast<unsigned char*>(hOutput)[j];hash*=1099511628211ULL;}
  printf("OUTPUT_HASH %016llx\n",(unsigned long long)hash);
  free(hRef);
  free(hInput);
  free(hOutput);
  check(platform.finalize());
  return maxDiff <= 1 ? 0 : 1;
}
