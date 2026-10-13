/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <memory>
#include <chrono>
#include <dirent.h>
#include <cstring>
#include "params.h"
#include "verification.h"
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


static int device=-1;
void generateVoxels(const float* points,size_t points_size,int repeat) {
  Params p;
  if(!points_size || points_size>MAX_POINTS_NUM)throw std::runtime_error("invalid point count");
  unsigned real=0;
  int gx=p.getGridXSize(),gy=p.getGridYSize(),gz=p.getGridZSize();
  size_t capacity=size_t(p.max_voxels)*p.max_points_per_voxel*p.feature_num;
  std::vector<unsigned> table(MAX_POINTS_NUM*4,UINT32_MAX),num(p.max_voxels,0),indices(p.max_voxels*4,0);
  std::vector<float> temp(capacity);std::memset(temp.data(),0xff,temp.size()*sizeof(float));
  std::vector<uint16_t> features(capacity,0);
  iris::DMem in(const_cast<float*>(points),points_size*p.feature_num*sizeof(float));
  iris::DMem hashmem(table.data(),table.size()*sizeof(unsigned)),nummem(num.data(),num.size()*sizeof(unsigned));
  iris::DMem tempmem(temp.data(),temp.size()*sizeof(float)),indexmem(indices.data(),indices.size()*sizeof(unsigned));
  iris::DMem realmem(&real,sizeof(real)),featuresmem(features.data(),features.size()*sizeof(uint16_t));
  size_t gws=((points_size+255)/256)*256,lws=256;
  auto start=std::chrono::steady_clock::now();
  for(int i=0;i<repeat;++i) {
    void* ba[]={&in,&points_size,arg(p.min_x_range),arg(p.max_x_range),arg(p.min_y_range),arg(p.max_y_range),arg(p.min_z_range),arg(p.max_z_range),arg(p.pillar_x_size),arg(p.pillar_y_size),arg(p.pillar_z_size),arg(gz),arg(gy),arg(gx),arg(p.feature_num),&hashmem,&realmem};
    int bi[]={iris_r,sizeof(points_size),sizeof(p.min_x_range),sizeof(p.max_x_range),sizeof(p.min_y_range),sizeof(p.max_y_range),sizeof(p.min_z_range),sizeof(p.max_z_range),sizeof(p.pillar_x_size),sizeof(p.pillar_y_size),sizeof(p.pillar_z_size),sizeof(gz),sizeof(gy),sizeof(gx),sizeof(p.feature_num),iris_rw,iris_rw};
    iris::Task build;check(build.kernel("buildHashKernel",1,nullptr,&gws,&lws,17,ba,bi));check(build.submit(device,nullptr,true));
    void* va[]={&in,&points_size,arg(p.min_x_range),arg(p.max_x_range),arg(p.min_y_range),arg(p.max_y_range),arg(p.min_z_range),arg(p.max_z_range),arg(p.pillar_x_size),arg(p.pillar_y_size),arg(p.pillar_z_size),arg(gz),arg(gy),arg(gx),arg(p.feature_num),arg(p.max_voxels),arg(p.max_points_per_voxel),&hashmem,&nummem,&tempmem,&indexmem};
    int vi[]={iris_r,sizeof(points_size),sizeof(p.min_x_range),sizeof(p.max_x_range),sizeof(p.min_y_range),sizeof(p.max_y_range),sizeof(p.min_z_range),sizeof(p.max_z_range),sizeof(p.pillar_x_size),sizeof(p.pillar_y_size),sizeof(p.pillar_z_size),sizeof(gz),sizeof(gy),sizeof(gx),sizeof(p.feature_num),sizeof(p.max_voxels),sizeof(p.max_points_per_voxel),iris_r,iris_rw,iris_rw,iris_rw};
    iris::Task voxel;check(voxel.kernel("voxelizationKernel",1,nullptr,&gws,&lws,21,va,vi));check(voxel.submit(device,nullptr,true));
  }
  printf("Average execution time of the voxelization kernel: %f (us)\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()*1e6/repeat);
  iris::Task count;check(count.flush_out(realmem));check(count.submit(device,nullptr,true));
  std::cout<<"valid_num: "<<real<<std::endl;
  unsigned valid=std::min(real,p.max_voxels);
  start=std::chrono::steady_clock::now();
  if(valid)for(int i=0;i<repeat;++i) {
    size_t fgws=((valid+255)/256)*256;
    void* fa[]={&tempmem,&nummem,arg(p.max_points_per_voxel),arg(p.feature_num),&featuresmem,&valid};
    int fi[]={iris_rw,iris_rw,sizeof(p.max_points_per_voxel),sizeof(p.feature_num),iris_rw,sizeof(valid)};
    iris::Task extract;check(extract.kernel("featureExtractionKernel",1,nullptr,&fgws,&lws,6,fa,fi));check(extract.submit(device,nullptr,true));
  }
  printf("Average execution time of the feature extraction kernel: %f (us)\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()*1e6/repeat);
  if(std::getenv("HECBENCH_VOXEL_DUMP")){
    iris::Task flush;check(flush.flush_out(nummem));check(flush.flush_out(indexmem));check(flush.flush_out(featuresmem));
    check(flush.submit(device,nullptr,true));dump_voxels(valid,num.data(),indices.data(),features.data());
  }
}
bool hasEnding(std::string const &fullString, std::string const &ending)
{
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
  } else {
    return false;
  }
}

int getFolderFile(const char *path, std::vector<std::string>& files, const char *suffix = ".bin")
{
  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir(path)) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      std::string file = ent->d_name;
      if(hasEnding(file, suffix)){
        files.push_back(file.substr(0, file.length()-4));
      }
    }
    closedir(dir);
  } else {
    std::cout << "No such folder: " << path << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return EXIT_SUCCESS;
}

int loadData(const char *file, void **data, unsigned int *length)
{
  std::fstream dataFile(file, std::ifstream::in);

  if (!dataFile.is_open()) {
    std::cout << "Can't open files: "<< file<<std::endl;
    return -1;
  }

  unsigned int len = 0;
  dataFile.seekg (0, dataFile.end);
  len = dataFile.tellg();
  dataFile.seekg (0, dataFile.beg);

  char *buffer = new char[len];
  if (buffer==NULL) {
    std::cout << "Can't malloc buffer."<<std::endl;
    dataFile.close();
    std::exit(EXIT_FAILURE);
  }

  dataFile.read(buffer, len);
  dataFile.close();

  *data = (void*)buffer;
  *length = len;
  return 0;
}

static void help()
{
  std::cout << "Usage: \n    ./main data/test/  100\n";
  std::cout << "    Run voxelization with data under ./data/test/ for 100 times\n";
  std::exit(EXIT_SUCCESS);
}

int main(int argc, char **argv)
{
  if (argc < 3)
    help();

  const char *data_folder  = argv[1];
  const int repeat  = atoi(argv[2]);

  std::vector<std::string> files;
  getFolderFile(data_folder, files);

  std::cout << "Number of files: " << files.size() << std::endl;

  Params params;

  iris::Platform platform;check(platform.init(&argc,&argv,true));device=hip_device();
  for (const auto & file : files)
  {
    std::string dataFile = data_folder + file + ".bin";

    std::cout << "\n<<<<<<<<<<<" <<std::endl;
    std::cout << "load file: "<< dataFile <<std::endl;

    unsigned int length = 0;
    void *pc_data = NULL;

    loadData(dataFile.c_str() , &pc_data, &length);
    size_t points_num = length / (params.feature_num * sizeof(float)) ;
    std::cout << "find points num: " << points_num << std::endl;

    generateVoxels(static_cast<const float*>(pc_data),points_num,repeat);
    delete[] static_cast<char*>(pc_data);

    std::cout << ">>>>>>>>>>>" <<std::endl;
  }

  check(platform.finalize());
  return 0;
}
