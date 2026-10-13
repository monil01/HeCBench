/***********************************************************************************
  Copyright 2015  Hung-Yi Pu, Kiyun Yun, Ziri Younsi, Sunk-Jin Yoon
  Odyssey  version 1.0   (released  2015)
  This file is part of Odyssey source code. Odyssey is a public, GPU-based code 
  for General Relativistic Radiative Transfer (GRRT), following the 
  ray-tracing algorithm presented in 
  Fuerst, S. V., & Wu, K. 2007, A&A, 474, 55, 
  and the radiative transfer formulation described in 
  Younsi, Z., Wu, K., & Fuerst, S. V. 2012, A&A, 545, A13

  Odyssey is distributed freely under the GNU general public license. 
  You can redistribute it and/or modify it under the terms of the License

  http://www.gnu.org/licenses/gpl.txt
  The current distribution website is:
  https://github.com/hungyipu/Odyssey/ 

 ***********************************************************************************/

#include <fstream>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <iris/iris.hpp>
#include <stdexcept>
#include "constants.h"



int main()
{
  // a set of variables defined in constants.h
  double  VariablesIn[VarINNUM];

  A           = 0.;    // black hole spin
  INCLINATION = acos(0.25)/PI*180.;     // inclination angle in unit of degree                    
  SIZE        = IMAGE_SIZE;
  printf("task1: image size = %d  x  %d  pixels\n",IMAGE_SIZE,IMAGE_SIZE);

  // number of grids; the coordinate of each grid is given by (GridIdxX,GridIdY)
  int ImaDimX, ImaDimY; 

  // number of blocks; the coordinate of each block is given by (blockIdx.x ,blockIdx.y )
  int GridDimX, GridDimY;

  // number of threads; the coordinate of each thread is given by (threadIdx.x,threadIdx.y)
  int BlockDimX, BlockDimY;

  // device buffers


  // save output results in files
  double* Results;
  FILE *fp;
  Results = new double[IMAGE_SIZE * IMAGE_SIZE * 3];

  BlockDimX = 100;
  BlockDimY = 1;
  GridDimX  = 1;
  GridDimY  = 50;
  size_t gws[] = {size_t(BlockDimX * GridDimX), size_t(BlockDimY * GridDimY)};
  size_t lws[] = {size_t(BlockDimX), size_t(BlockDimY)};

  //compute number of grides, to cover the whole image plane
  ImaDimX = (int)ceil((double)IMAGE_SIZE / (BlockDimX * GridDimX));
  ImaDimY = (int)ceil((double)IMAGE_SIZE / (BlockDimY * GridDimY));

  iris::Platform platform;
  int argc=0; char** argv=nullptr;
  if(platform.init(&argc,&argv,1)) return 2;
  int count=0, device=-1; iris_device_count(&count);
  for(int i=0;i<count;++i) {
    int backend=0,type=0; char name[256]={};
    iris_device_info(i,iris_backend,&backend,nullptr);
    iris_device_info(i,iris_type,&type,nullptr);
    iris_device_info(i,iris_name,name,nullptr);
    fprintf(stderr,"IRIS device %d backend=%d type=%d name=%s\n",i,backend,type,name);
    if(backend==3 && type==256) device=i;
  }
  if(device<0) return 2;
  iris::DMem results(Results,sizeof(double)*IMAGE_SIZE*IMAGE_SIZE*3);
  iris::DMem variables(VariablesIn,sizeof(VariablesIn));
  auto flush=[&]() { iris::Task task; task.flush_out(results);
    if(task.submit(device,nullptr,1)) throw std::runtime_error("flush failed"); };
  auto launch=[&](const char* name,int x,int y) {
    iris::Task task; void* params[]={&results,&variables,&x,&y};
    int info[]={iris_rw,iris_r,sizeof(x),sizeof(y)};
    if(task.kernel(name,2,nullptr,gws,lws,4,params,info) ||
       task.submit(device,nullptr,1)) throw std::runtime_error("kernel failed");
  };
  variables.update(VariablesIn);

  auto start = std::chrono::steady_clock::now();

  for(int GridIdxY = 0; GridIdxY < ImaDimY; GridIdxY++){
    for(int GridIdxX = 0; GridIdxX < ImaDimX; GridIdxX++){                      
      launch("task1",GridIdxX,GridIdxY);
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (task1) %f (s)\n", time * 1e-9f);
  

  flush();

  //save result to output
  fp=fopen("Output_task1.txt","w");  
  if (fp != NULL) {
    fprintf(fp,"###output data:(alpha,  beta,  redshift)\n");

    for(int j = 0; j < IMAGE_SIZE; j++)
      for(int i = 0; i < IMAGE_SIZE; i++)
      {
        fprintf(fp, "%f\t", (float)Results[3 * (IMAGE_SIZE * j + i) + 0]);
        fprintf(fp, "%f\t", (float)Results[3 * (IMAGE_SIZE * j + i) + 1]);
        fprintf(fp, "%f\n", (float)Results[3 * (IMAGE_SIZE * j + i) + 2]);
      }
    fclose(fp);
  }

  A           = 0.;    // black hole spin
  INCLINATION = 45.;   // inclination angle in unit of degree                    
  SIZE        = IMAGE_SIZE;
  freq_obs    = 340e9; // observed frequency
  printf("task2: image size = %d  x  %d  pixels\n",IMAGE_SIZE,IMAGE_SIZE);

  variables.update(VariablesIn);

  start = std::chrono::steady_clock::now();

  //compute number of grides, to cover the whole image plane
  for(int GridIdxY = 0; GridIdxY < ImaDimY; GridIdxY++){
    for(int GridIdxX = 0; GridIdxX < ImaDimX; GridIdxX++){                      
      launch("task2",GridIdxX,GridIdxY);
    }
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total kernel execution time (task2) %f (s)\n", time * 1e-9f);

  flush();

  fp=fopen("Output_task2.txt","w");  
  if (fp != NULL) {
    fprintf(fp,"###output data:(alpha,  beta, Luminosity (erg/sec))\n");

    for(int j = 0; j < IMAGE_SIZE; j++)
      for(int i = 0; i < IMAGE_SIZE; i++)
      {
        fprintf(fp, "%f\t", (float)Results[3 * (IMAGE_SIZE * j + i) + 0]);
        fprintf(fp, "%f\t", (float)Results[3 * (IMAGE_SIZE * j + i) + 1]);
        fprintf(fp, "%f\n", (float)Results[3 * (IMAGE_SIZE * j + i) + 2]);
      }
    fclose(fp);
  }

  platform.finalize();
  delete [] Results;
  return 0;
}
