/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iris/iris.hpp>
#include <stdexcept>
static void check(int rc){if(rc!=IRIS_SUCCESS)throw std::runtime_error("IRIS API failure");}
template<class T> static void* arg(const T& value){return const_cast<T*>(&value);}
static int hip_device(){
  int n=0;check(iris_device_count(&n));
  for(int d=0;d<n;++d){int backend=0,type=0;char name[256]={};
    check(iris_device_info(d,iris_backend,&backend,nullptr));check(iris_device_info(d,iris_type,&type,nullptr));
    check(iris_device_info(d,iris_name,name,nullptr));
    printf("IRIS device %d: %s backend=%d type=%d\n",d,name,backend,type);
    if(backend==iris_hip&&(type&iris_gpu))return d;
  }
  throw std::runtime_error("no IRIS HIP GPU");
}

#include "params.h"
#include "reference.h"

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  Params p;  // constant values defined in params.h
  const float min_x_range = p.min_x_range;
  const float max_x_range = p.max_x_range;
  const float min_y_range = p.min_y_range;
  const float max_y_range = p.max_y_range;
  const int feature_x_size = p.feature_x_size;
  const int feature_y_size = p.feature_y_size;
  const int num_anchors = p.num_anchors;
  const int num_classes = p.num_classes;
  const int num_box_values = p.num_box_values;
  const float score_thresh = p.score_thresh;
  const float dir_offset = p.dir_offset;
  const int len_per_anchor = p.len_per_anchor;
  const int num_dir_bins = p.num_dir_bins;

  const int feature_size = feature_x_size * feature_y_size;
  const int feature_anchor_size = feature_size * num_anchors;
  const int cls_size = feature_anchor_size * num_classes;
  const int box_size = feature_anchor_size * num_box_values;
  const int dir_cls_size = feature_anchor_size * num_dir_bins;
  const int bndbox_size = feature_anchor_size * 9;

  const int cls_size_byte = cls_size * sizeof(float);
  const int box_size_byte = box_size * sizeof(float);
  const int dir_cls_size_byte = dir_cls_size * sizeof(float);
  const int bndbox_size_byte = bndbox_size * sizeof(float);
  const int score_size_byte = feature_anchor_size * sizeof(float);

  // input of the post-process kernel
  float *h_cls_input = (float*) malloc (cls_size_byte);
  float *h_box_input = (float*) malloc (box_size_byte);
  float *h_dir_cls_input = (float*) malloc (dir_cls_size_byte);

  // output of the post-process kernel
  float *h_bndbox_output = (float*) malloc (bndbox_size_byte);
  float *h_score_output = (float*) malloc (score_size_byte);

  // random values
  srand(123);
  for (int i = 0; i < cls_size; i++)  h_cls_input[i] = rand() / (float)RAND_MAX;
  for (int i = 0; i < box_size; i++)  h_box_input[i] = rand() / (float)RAND_MAX;
  for (int i = 0; i < dir_cls_size; i++)  h_dir_cls_input[i] = rand() / (float)RAND_MAX;

  iris::Platform platform;check(platform.init(&argc,&argv,true));int device=hip_device();
  int bndbox_num=0;
  iris::DMem d_cls_input(h_cls_input,cls_size_byte),d_box_input(h_box_input,box_size_byte);
  iris::DMem d_dir_cls_input(h_dir_cls_input,dir_cls_size_byte),d_bndbox_output(h_bndbox_output,bndbox_size_byte);
  iris::DMem d_score_output(h_score_output,score_size_byte),d_object_counter(&bndbox_num,sizeof(int));
  iris::DMem d_anchors(const_cast<float*>(p.anchors),num_anchors*len_per_anchor*sizeof(float));
  iris::DMem d_anchor_bottom_heights(const_cast<float*>(p.anchor_bottom_heights),num_classes*sizeof(float));
  double time = 0.0;

  size_t gws=size_t(feature_size)*num_anchors,lws=num_anchors;

  for (int i = 0; i < repeat; i++) {
    check(d_box_input.update(h_box_input));
    bndbox_num=0;check(d_object_counter.update(&bndbox_num));

    auto start = std::chrono::steady_clock::now();

    void* args[]={&d_cls_input,&d_box_input,&d_dir_cls_input,&d_anchors,&d_anchor_bottom_heights,&d_bndbox_output,&d_score_output,&d_object_counter,arg(min_x_range),arg(max_x_range),arg(min_y_range),arg(max_y_range),arg(feature_x_size),arg(feature_y_size),arg(num_anchors),arg(num_classes),arg(num_box_values),arg(score_thresh),arg(dir_offset)};
    int info[]={iris_r,iris_rw,iris_r,iris_r,iris_r,iris_w,iris_w,iris_rw,sizeof(min_x_range),sizeof(max_x_range),sizeof(min_y_range),sizeof(max_y_range),sizeof(feature_x_size),sizeof(feature_y_size),sizeof(num_anchors),sizeof(num_classes),sizeof(num_box_values),sizeof(score_thresh),sizeof(dir_offset)};
    iris::Task task;
    check(task.kernel("postprocess",1,nullptr,&gws,&lws,19,args,info));
    check(task.submit(device,nullptr,true));
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  printf("Average execution time of postprocess kernel: %f (us)\n", (time * 1e-3f) / repeat);

  iris::Task flush;check(flush.flush_out(d_object_counter));check(flush.flush_out(d_bndbox_output));
  check(flush.flush_out(d_score_output));check(flush.submit(device,nullptr,true));
  printf("Detected boxes: %d\n",bndbox_num);
  verify (bndbox_num,
          h_cls_input,
          h_box_input,
          h_dir_cls_input,
          p.anchors,
          p.anchor_bottom_heights,
          h_bndbox_output,
          h_score_output,
          min_x_range,
          max_x_range,
          min_y_range,
          max_y_range,
          feature_size,
          feature_x_size,
          feature_y_size,
          num_anchors,
          num_classes,
          num_box_values,
          score_thresh,
          dir_offset);


  free(h_cls_input);
  free(h_box_input);
  free(h_dir_cls_input);
  free(h_score_output);
  free(h_bndbox_output);

  check(platform.finalize());
  return 0;
}
