#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdint.h>
#include <chrono>
#include <utility>  // std::swap
#include "iris_host.h"
#include "utils.h"
#include "config.h"
#include "kernels_wrapper.h"

//#define STBI_ONLY_BMP
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"




int main(int argc, char **argv) {
  if(argc < 3){
    printf("Usage: %s <file> <number of seams to remove> [options]\n"
           "valid options:\n-u\tupdate costs instead of recomputing them\n"
           "-a\tapproximate computation\n", argv[0]);
    return 1;
  }

  iris::Platform platform;if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS) return 1;
  int backend=0,type=0;char name[256]={};
  iris_device_info(0,iris_backend,&backend,nullptr);iris_device_info(0,iris_type,&type,nullptr);
  iris_device_info(0,iris_name,name,nullptr);
  printf("Device backend=%d type=%d name=%s\n",backend,type,name);
  if(backend!=3 || type!=256) return 1;
  char *check;
  long seams_to_remove = strtol(argv[2], &check, 10);  //10 specifies base-10
  if (check == argv[2]){   //if no characters were converted pointers are equal
    printf("ERROR: can't convert string to number, exiting.\n");
    return 1;
  }

  int w, h, ncomp;
  unsigned char* imgv = stbi_load(argv[1], &w, &h, &ncomp, 0);
  if(imgv == NULL){
    printf("ERROR: can't load image \"%s\", exiting.\n", argv[1]);
    printf("Reason: %s\n", stbi_failure_reason());
    return 1;
  }

  if(ncomp != 3){
    printf("ERROR: image does not have 3 components (RGB), exiting.\n");
    return 1;
  }

  if(seams_to_remove < 0 || seams_to_remove >= w){
    printf("ERROR: number of seams to remove is invalid, exiting.\n");
    return 1;
  }

  seam_carver_mode mode = SEAM_CARVER_STANDARD_MODE;

  if(argc >= 4){
    if(strcmp(argv[3],"-u") == 0){
      mode = SEAM_CARVER_UPDATE_MODE;
      printf("update mode selected.\n");
    }
    else if(strcmp(argv[3],"-a") == 0){
      mode = SEAM_CARVER_APPROX_MODE;
      printf("approximation mode selected.\n");
    }
    else{
      printf("an invalid option was specified and will be ignored. Valid options are: -u, -a.\n");
    }
  }

  printf("Image loaded. Resizing...\n");

  int current_w = w;
  uchar4 *h_pixels = build_pixels(imgv, w, h);
  const size_t img_bytes = (size_t)w * h * sizeof(uchar4);
  const int cost_bytes = w * h * sizeof(short);
  const int h_bytes = h * sizeof(int);
  const int w_bytes = w * sizeof(int);

  int* indices=(int*)malloc(w_bytes);
  for(int i=0;i<w;i++) indices[i]=i;
  iris::DMem mem_d_pixels(h_pixels,img_bytes);
  iris::DMem* d_pixels=&mem_d_pixels;
  std::vector<uchar4> host_d_pixels_swap(w*h);
  iris::DMem mem_d_pixels_swap(host_d_pixels_swap.data(),img_bytes);
  iris::DMem* d_pixels_swap=&mem_d_pixels_swap;
  std::vector<short> host_d_costs_left(w*h);
  iris::DMem mem_d_costs_left(host_d_costs_left.data(),cost_bytes);
  iris::DMem* d_costs_left=&mem_d_costs_left;
  std::vector<short> host_d_costs_up(w*h);
  iris::DMem mem_d_costs_up(host_d_costs_up.data(),cost_bytes);
  iris::DMem* d_costs_up=&mem_d_costs_up;
  std::vector<short> host_d_costs_right(w*h);
  iris::DMem mem_d_costs_right(host_d_costs_right.data(),cost_bytes);
  iris::DMem* d_costs_right=&mem_d_costs_right;
  std::vector<short> host_d_costs_swap_left(w*h);
  iris::DMem mem_d_costs_swap_left(host_d_costs_swap_left.data(),cost_bytes);
  iris::DMem* d_costs_swap_left=&mem_d_costs_swap_left;
  std::vector<short> host_d_costs_swap_up(w*h);
  iris::DMem mem_d_costs_swap_up(host_d_costs_swap_up.data(),cost_bytes);
  iris::DMem* d_costs_swap_up=&mem_d_costs_swap_up;
  std::vector<short> host_d_costs_swap_right(w*h);
  iris::DMem mem_d_costs_swap_right(host_d_costs_swap_right.data(),cost_bytes);
  iris::DMem* d_costs_swap_right=&mem_d_costs_swap_right;
  std::vector<int> host_d_M(w*h);
  iris::DMem mem_d_M(host_d_M.data(),img_bytes);
  iris::DMem* d_M=&mem_d_M;
  std::vector<int> host_d_index_map(w*h);
  iris::DMem mem_d_index_map(host_d_index_map.data(),img_bytes);
  iris::DMem* d_index_map=&mem_d_index_map;
  std::vector<int> host_d_offset_map(w*h);
  iris::DMem mem_d_offset_map(host_d_offset_map.data(),img_bytes);
  iris::DMem* d_offset_map=&mem_d_offset_map;
  std::vector<int> host_d_indices(w);
  iris::DMem mem_d_indices(host_d_indices.data(),w_bytes);
  iris::DMem* d_indices=&mem_d_indices;
  iris::DMem mem_d_indices_ref(indices,w_bytes);
  iris::DMem* d_indices_ref=&mem_d_indices_ref;
  std::vector<int> host_d_seam(h);
  iris::DMem mem_d_seam(host_d_seam.data(),h_bytes);
  iris::DMem* d_seam=&mem_d_seam;
  iris::DMem* reduce_row=d_M;
  int reduce_offset=(mode==SEAM_CARVER_APPROX_MODE)?0:w*(h-1);
  if(mode == SEAM_CARVER_UPDATE_MODE)
    compute_costs(current_w, w, h, d_pixels, d_costs_left, d_costs_up, d_costs_right);

  int num_iterations = 0;

  auto start = std::chrono::steady_clock::now();

  while(num_iterations < (int)seams_to_remove){

    if(mode == SEAM_CARVER_STANDARD_MODE)
      compute_costs(current_w, w, h, d_pixels, d_costs_left, d_costs_up, d_costs_right);

    if(mode != SEAM_CARVER_APPROX_MODE){
      compute_M(current_w, w, h, d_M, d_costs_left, d_costs_up, d_costs_right);
      find_min_index(current_w, d_indices_ref, d_indices, reduce_row, reduce_offset);
      find_seam(current_w, w, h, d_M, d_indices, d_seam);
    }
    else{
      approx_setup(current_w, w, h, d_pixels, d_index_map, d_offset_map, d_M);
      approx_M(current_w, w, h, d_offset_map,  d_M);
      find_min_index(current_w, d_indices_ref, d_indices, reduce_row, reduce_offset);
      approx_seam(w, h, d_index_map, d_indices, d_seam);
    }

    remove_seam(current_w, w, h, d_M, d_pixels, d_pixels_swap, d_seam);
    std::swap(d_pixels, d_pixels_swap);

    if(mode == SEAM_CARVER_UPDATE_MODE){
      update_costs(current_w, w, h, d_M, d_pixels,
                   d_costs_left, d_costs_up, d_costs_right,
                   d_costs_swap_left, d_costs_swap_up, d_costs_swap_right, d_seam );
      std::swap(d_costs_left, d_costs_swap_left);
      std::swap(d_costs_up, d_costs_swap_up);
      std::swap(d_costs_right, d_costs_swap_right);
    }

    current_w--;
    num_iterations++;
  }


  auto end = std::chrono::steady_clock::now();
  float time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Execution time of seam carver kernels: %f (ms)\n", time * 1e-6f);

  if(d_pixels->fetch(h_pixels)!=IRIS_SUCCESS) return 1;
  unsigned char* output = flatten_pixels(h_pixels, w, h, current_w);
  printf("Image resized\n");

  printf("Saving in resized.bmp...\n");
  int success = stbi_write_bmp("resized.bmp", current_w, h, 3, output);
  printf("%s\n", success ? "Success" : "Failed");
  printf("%s\n", success ? "PASS" : "FAIL");

  if(mode != SEAM_CARVER_APPROX_MODE){
        }
  if(mode == SEAM_CARVER_UPDATE_MODE){
        }
  if(mode == SEAM_CARVER_APPROX_MODE){
      }
  free(h_pixels);
  free(output);
  free(indices);
  free(imgv);
  platform.finalize();
  return success?0:1;
}

