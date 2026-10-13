#include <iris/iris.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <chrono>
#include <utility>
#include "iris_host.h"
struct alignas(4) uchar4 {unsigned char x,y,z,w;};
static_assert(sizeof(uchar4)==4,"uchar4 ABI");
#include "tables.h"
#include "params.h"

int main(int argc,char** argv) {
 if(argc!=2) {printf("Usage: %s <repeat>\n",argv[0]);return 1;}
 unsigned int repeat=atoi(argv[1]);
 iris::Platform platform;if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS) return 1;
 int backend=0,type=0;char name[256]={};
 iris_device_info(0,iris_backend,&backend,nullptr);iris_device_info(0,iris_type,&type,nullptr);iris_device_info(0,iris_name,name,nullptr);
 printf("Device backend=%d type=%d name=%s\n",backend,type,name);
 if(backend!=3 || type!=256 || !strstr(name,"AMD")) return 1;
 std::uniform_real_distribution<float>rd(0,1);std::mt19937 mt(123);
 iris::DMem minMaxLv1Device(blockNum*2*sizeof(float)),blockIndicesLv1Device(blockNum*sizeof(unsigned int));
 unsigned int countedBlockNumLv1=0,countedBlockNumLv2=0,countedVerticesNum=0,countedTrianglesNum=0;
 iris::DMem countedBlockNumLv1Device(&countedBlockNumLv1,sizeof(unsigned int)),countedBlockNumLv2Device(&countedBlockNumLv2,sizeof(unsigned int));
 iris::DMem countedVerticesNumDevice(&countedVerticesNum,sizeof(unsigned int)),countedTrianglesNumDevice(&countedTrianglesNum,sizeof(unsigned int));
 iris::DMem distinctEdgesTableDevice(distinctEdgesTable,sizeof(distinctEdgesTable)),triTableDevice(triTable,sizeof(triTable)),edgeIDTableDevice(edgeIDTable,sizeof(edgeIDTable));
 unsigned long long triangleSum=0;float cx=0,cy=0,cz=0,zp=0;
 iris::DMem trianglesDevice(&triangleSum,sizeof(triangleSum)),coordXDevice(&cx,sizeof(float)),coordYDevice(&cy,sizeof(float)),coordZDevice(&cz,sizeof(float)),coordZPDevice(&zp,sizeof(float));
 const Dim BlockSizeLv1(voxelXLv1,voxelYLv1,1),GridSizeLv1(gridXLv1,gridYLv1,gridZLv1);
 const Dim BlockSizeLv2(voxelXLv2*voxelYLv2,blockXLv2*blockYLv2,1),BlockSizeGenerating(voxelXLv2,voxelYLv2,voxelZLv2);
 float isoValue=-0.9f,time=0.f;
 for(unsigned int c0=0;c0<repeat;++c0) {
  countedBlockNumLv1=countedBlockNumLv2=countedVerticesNum=countedTrianglesNum=0;
  triangleSum=0;cx=cy=cz=zp=0;
  countedBlockNumLv1Device.update(&countedBlockNumLv1);countedBlockNumLv2Device.update(&countedBlockNumLv2);
  countedVerticesNumDevice.update(&countedVerticesNum);countedTrianglesNumDevice.update(&countedTrianglesNum);
  trianglesDevice.update(&triangleSum);coordXDevice.update(&cx);coordYDevice.update(&cy);coordZDevice.update(&cz);coordZPDevice.update(&zp);
  launch("iris_computeMinMaxLv1",GridSizeLv1,BlockSizeLv1,minMaxLv1Device);
  launch("iris_compactLv1",Dim(countingBlockNumLv1),Dim(countingThreadNumLv1),isoValue,std::as_const(minMaxLv1Device),blockIndicesLv1Device,countedBlockNumLv1Device);
  flush(countedBlockNumLv1Device);
  iris::DMem minMaxLv2Device(size_t(countedBlockNumLv1)*voxelNumLv2*2*sizeof(float));
  launch("iris_computeMinMaxLv2",Dim(countedBlockNumLv1),BlockSizeLv2,std::as_const(blockIndicesLv1Device),minMaxLv2Device);
  iris::DMem blockIndicesLv2Device(size_t(countedBlockNumLv1)*voxelNumLv2*sizeof(unsigned int));
  unsigned int countingBlockNumLv2=(countedBlockNumLv1*voxelNumLv2+countingThreadNumLv2-1)/countingThreadNumLv2;
  launch("iris_compactLv2",Dim(countingBlockNumLv2),Dim(countingThreadNumLv2),isoValue,std::as_const(minMaxLv2Device),std::as_const(blockIndicesLv1Device),blockIndicesLv2Device,countedBlockNumLv1,countedBlockNumLv2Device);
  flush(countedBlockNumLv2Device);
  auto start=std::chrono::steady_clock::now();
  launch("iris_generatingTriangles",Dim(countedBlockNumLv2),BlockSizeGenerating,isoValue,std::as_const(blockIndicesLv2Device),std::as_const(distinctEdgesTableDevice),std::as_const(triTableDevice),std::as_const(edgeIDTableDevice),countedVerticesNumDevice,countedTrianglesNumDevice,trianglesDevice,coordXDevice,coordYDevice,coordZDevice,coordZPDevice);
  auto end=std::chrono::steady_clock::now();
  time+=std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  flush(countedVerticesNumDevice,countedTrianglesNumDevice);
 }
  printf("Block Lv1: %u\nBlock Lv2: %u\n", countedBlockNumLv1, countedBlockNumLv2);
  printf("Vertices Size: %u\n", countedBlockNumLv2 * 304);
  printf("Triangles Size: %u\n", countedBlockNumLv2 * 315 * 3);
  printf("Vertices: %u\nTriangles: %u\n", countedVerticesNum, countedTrianglesNum);
  printf("Average kernel execution time (generatingTriangles): %f (s)\n", (time * 1e-9f) / repeat);

  // specific to the problem size
  bool ok = (countedBlockNumLv1 == 8296 && countedBlockNumLv2 == 240380 &&
             countedVerticesNum == 4856560 && countedTrianglesNum == 6101640);
  printf("%s\n", ok ? "PASS" : "FAIL");

  platform.finalize();return 0;
}
