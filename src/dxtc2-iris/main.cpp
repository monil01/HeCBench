/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Utilities and system includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "dmem.h"
#include <algorithm>
using uint=unsigned int;

#include <float.h>  // for FLT_MAX
#include <chrono>


#include "dds.h"
#include "permutations.h"
#include "shrUtils.h"

#define ERROR_THRESHOLD 0.02f

#define NUM_THREADS 64  // Number of threads per block.

// Helper structs and functions to validate the output of the compressor.
// We cannot simply do a bitwise compare, because different compilers produce
// different
// results for different targets due to floating point arithmetic.

union Color32 {
  struct {
    unsigned char b, g, r, a;
  };
  unsigned int u;
};

union Color16 {
  struct {
    unsigned short b : 5;
    unsigned short g : 6;
    unsigned short r : 5;
  };
  unsigned short u;
};

struct BlockDXT1 {
  Color16 col0;
  Color16 col1;
  union {
    unsigned char row[4];
    unsigned int indices;
  };

  void decompress(Color32 colors[16]) const;
};

void BlockDXT1::decompress(Color32 *colors) const {
  Color32 palette[4];

  // Does bit expansion before interpolation.
  palette[0].b = (col0.b << 3) | (col0.b >> 2);
  palette[0].g = (col0.g << 2) | (col0.g >> 4);
  palette[0].r = (col0.r << 3) | (col0.r >> 2);
  palette[0].a = 0xFF;

  palette[1].r = (col1.r << 3) | (col1.r >> 2);
  palette[1].g = (col1.g << 2) | (col1.g >> 4);
  palette[1].b = (col1.b << 3) | (col1.b >> 2);
  palette[1].a = 0xFF;

  if (col0.u > col1.u) {
    // Four-color block: derive the other two colors.
    palette[2].r = (2 * palette[0].r + palette[1].r) / 3;
    palette[2].g = (2 * palette[0].g + palette[1].g) / 3;
    palette[2].b = (2 * palette[0].b + palette[1].b) / 3;
    palette[2].a = 0xFF;

    palette[3].r = (2 * palette[1].r + palette[0].r) / 3;
    palette[3].g = (2 * palette[1].g + palette[0].g) / 3;
    palette[3].b = (2 * palette[1].b + palette[0].b) / 3;
    palette[3].a = 0xFF;
  } else {
    // Three-color block: derive the other color.
    palette[2].r = (palette[0].r + palette[1].r) / 2;
    palette[2].g = (palette[0].g + palette[1].g) / 2;
    palette[2].b = (palette[0].b + palette[1].b) / 2;
    palette[2].a = 0xFF;

    palette[3].r = 0x00;
    palette[3].g = 0x00;
    palette[3].b = 0x00;
    palette[3].a = 0x00;
  }

  for (int i = 0; i < 16; i++) {
    colors[i] = palette[(indices >> (2 * i)) & 0x3];
  }
}

static int compareColors(const Color32 *b0, const Color32 *b1) {
  int sum = 0;

  for (int i = 0; i < 16; i++) {
    int r = (b0[i].r - b1[i].r);
    int g = (b0[i].g - b1[i].g);
    int b = (b0[i].b - b1[i].b);
    sum += r * r + g * g + b * b;
  }

  return sum;
}

static int compareBlock(const BlockDXT1 *b0, const BlockDXT1 *b1) {
  Color32 colors0[16];
  Color32 colors1[16];

  if (memcmp(b0, b1, sizeof(BlockDXT1)) == 0) {
    return 0;
  } else {
    b0->decompress(colors0);
    b1->decompress(colors1);

    return compareColors(colors0, colors1);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  if(argc!=4)return 1;
  iris::Platform platform;if(platform.init(&argc,&argv,true)!=IRIS_SUCCESS)return 1;
  int dev=require_amd_hip();if(dev<0)return 1;iris_device_set_default(dev);

  // Load input image.
  unsigned char *data = NULL;
  uint w, h;

  const char *image_path = argv[1];

  const char *reference_image_path = argv[2];

  const int numIterations = atoi(argv[3]);

  if (!shrLoadPPM4ub(image_path, (unsigned char **)&data, &w, &h)) {
    printf("Error: unable to open source image file <%s>\n", image_path);
    exit(EXIT_FAILURE);
  }

  if (w % 4 != 0 || h % 4 != 0) {
    printf("Error: the image dimensions must be a multiple of 4.\n");
    free(data);
    exit(EXIT_FAILURE);
  }

  if (w > 16384 || h > 16384) {
    printf("Error: the image dimensions exceed the maximum values.\n");
    free(data);
    exit(EXIT_FAILURE);
  }

  printf("Image Loaded '%s', %d x %d pixels\n\n", image_path, w, h);

  // Allocate input image.
  const uint memSize = w * h * 4;
  assert(0 != memSize);
  uint *block_image = (uint *)malloc(memSize);

  // Convert linear image to block linear.
  for (uint by = 0; by < h / 4; by++) {
    for (uint bx = 0; bx < w / 4; bx++) {
      for (int i = 0; i < 16; i++) {
        const int x = i & 3;
        const int y = i / 4;
        block_image[(by * w / 4 + bx) * 16 + i] =
            ((uint *)data)[(by * 4 + y) * 4 * (w / 4) + bx * 4 + x];
      }
    }
  }

  // copy into global mem
  void* d_data = NULL;
  allocDeviceBuffer(&d_data,memSize);

  // Result
  void* d_result = NULL;
  const uint compressedSize = (w / 4) * (h / 4) * 8;
  allocDeviceBuffer(&d_result,compressedSize);
  uint *h_result = (uint *)malloc(compressedSize);

  // Compute permutations.
  uint permutations[1024];
  computePermutations(permutations);

  // Copy permutations host to devie.
  void* d_permutations = NULL;
  allocDeviceBuffer(&d_permutations,1024 * sizeof(uint));
  copyToDevice(d_permutations, permutations, 1024 * sizeof(uint));

  // Copy image from host to device
  copyToDevice(d_data, block_image, memSize);

  // Determine launch configuration and run timed computation numIterations
  // times
  uint blocks = w / 4 * h / 4;

  // Restrict the numbers of blocks to launch on low end GPUs to avoid kernel
  // timeout
  //int blocksPerLaunch = min(blocks, 768 * deviceProp.multiProcessorCount);
  int blocksPerLaunch = MIN(blocks, 768 * 24);

  printf("Running DXT Compression on %u x %u image...\n", w, h);
  printf("\n%u Blocks, %u Threads per Block, %u Threads in Grid...\n\n", blocks,
         NUM_THREADS, blocks * NUM_THREADS);

  

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < numIterations; ++i) {
    for (int j = 0; j < (int)blocks; j += blocksPerLaunch) {
      launch("compress",std::min<unsigned>(blocksPerLaunch,blocks-j),NUM_THREADS,{memory(d_permutations),memory(d_data),memory(d_result),scalar(j)});
    }
  }

  // sync to host
  
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / numIterations);

  // copy result data from device to host
  copyFromDevice(h_result, d_result, compressedSize);

  // Write out result data to DDS file
  char output_filename[1024];
  strcpy(output_filename, image_path);
  strcpy(output_filename + strlen(image_path) - 3, "dds");
  FILE *fp = fopen(output_filename, "wb");

  if (fp == 0) {
    printf("Error, unable to open output image <%s>\n", output_filename);
    exit(EXIT_FAILURE);
  }

  DDSHeader header;
  header.fourcc = FOURCC_DDS;
  header.size = 124;
  header.flags = (DDSD_WIDTH | DDSD_HEIGHT | DDSD_CAPS | DDSD_PIXELFORMAT |
                  DDSD_LINEARSIZE);
  header.height = h;
  header.width = w;
  header.pitch = compressedSize;
  header.depth = 0;
  header.mipmapcount = 0;
  memset(header.reserved, 0, sizeof(header.reserved));
  header.pf.size = 32;
  header.pf.flags = DDPF_FOURCC;
  header.pf.fourcc = FOURCC_DXT1;
  header.pf.bitcount = 0;
  header.pf.rmask = 0;
  header.pf.gmask = 0;
  header.pf.bmask = 0;
  header.pf.amask = 0;
  header.caps.caps1 = DDSCAPS_TEXTURE;
  header.caps.caps2 = 0;
  header.caps.caps3 = 0;
  header.caps.caps4 = 0;
  header.notused = 0;
  fwrite(&header, sizeof(DDSHeader), 1, fp);
  fwrite(h_result, compressedSize, 1, fp);
  fclose(fp);

  fp = fopen(reference_image_path, "rb");

  if (fp == 0) {
    printf("Error, unable to open reference image\n");

    exit(EXIT_FAILURE);
  }

  fseek(fp, sizeof(DDSHeader), SEEK_SET);
  uint referenceSize = (w / 4) * (h / 4) * 8;
  uint *reference = (uint *)malloc(referenceSize);
  fread(reference, referenceSize, 1, fp);
  fclose(fp);

  printf("\nChecking accuracy...\n");
  float rms = 0;

  for (uint y = 0; y < h; y += 4) {
    for (uint x = 0; x < w; x += 4) {
      uint referenceBlockIdx = ((y / 4) * (w / 4) + (x / 4));
      uint resultBlockIdx = ((y / 4) * (w / 4) + (x / 4));

      int cmp = compareBlock(((BlockDXT1 *)h_result) + resultBlockIdx,
                             ((BlockDXT1 *)reference) + referenceBlockIdx);

      //if (cmp != 0.0f) {
      //  printf("Deviation at (%4d,%4d):\t%f rms\n", x / 4, y / 4,
      //         float(cmp) / 16 / 3);
      //}

      rms += cmp;
    }
  }

  rms /= w * h * 3;

  // Free allocated resources
  freeDeviceBuffer(d_permutations);
  freeDeviceBuffer(d_data);
  freeDeviceBuffer(d_result);
  free(data);
  free(block_image);
  free(h_result);
  free(reference);

  printf("RMS(reference, result) = %f\n\n", rms);
  printf(rms <= ERROR_THRESHOLD ? "PASS\n" : "FAIL\n");
  /* Return zero if test passed, one otherwise */
  int errors=platform.error_count();platform.finalize();return rms > ERROR_THRESHOLD || errors;
}
