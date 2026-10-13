/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/*
 * GPU accelerated coulombic potential grid test code
 */

#include <stdio.h>
#include <stdlib.h>
#include <iris/iris.hpp>
#include <cstring>
#include <cstdint>
#include <vector>
#include <stdexcept>
struct dim3 { unsigned x=1,y=1,z=1; };
struct float3 {float x,y,z;};
struct alignas(16) float4 {float x,y,z,w;};
#define MAXATOMS 4000
#define UNROLLX 8
#define UNROLLY 1
#define BLOCKSIZEX 8
#define BLOCKSIZEY 8
static void check(int rc){if(rc!=IRIS_SUCCESS) throw std::runtime_error("IRIS API failure");}
#include "WKFUtils.h"

int copyatoms(float *atoms, int count, float zplane, float4 *atominfo) {

  if (count > MAXATOMS) {
    printf("Atom count exceeds constant buffer storage capacity\n");
    return -1;
  }

  float atompre[4*MAXATOMS];
  int i;
  for (i=0; i<count*4; i+=4) {
    atompre[i    ] = atoms[i    ];
    atompre[i + 1] = atoms[i + 1];
    float dz = zplane - atoms[i + 2];
    atompre[i + 2]  = dz*dz;
    atompre[i + 3] = atoms[i + 3];
  }

  std::memcpy(atominfo, atompre, count * 4 * sizeof(float));

  return 0;
}


int initatoms(float **atombuf, int count, dim3 volsize, float gridspacing) {
  float3 size;
  int i;
  float *atoms;
  srand(2);

  atoms = (float *) malloc(count * 4 * sizeof(float));
  *atombuf = atoms;

  // compute grid dimensions in angstroms
  size.x = gridspacing * volsize.x;
  size.y = gridspacing * volsize.y;
  size.z = gridspacing * volsize.z;

  for (i=0; i<count; i++) {
    int addr = i * 4;
    atoms[addr    ] = (rand() / (float) RAND_MAX) * size.x;
    atoms[addr + 1] = (rand() / (float) RAND_MAX) * size.y;
    atoms[addr + 2] = (rand() / (float) RAND_MAX) * size.z;
    atoms[addr + 3] = ((rand() / (float) RAND_MAX) * 2.0) - 1.0;  // charge
  }

  return 0;
}


int main(int argc, char** argv) {
  float *energy = NULL;
  float *atoms = NULL;
  dim3 volsize, Gsz, Bsz;
  wkf_timerhandle runtimer, mastertimer, copytimer, hostcopytimer;
  float copytotal, runtotal, mastertotal, hostcopytotal;
  const char *statestr = "|/-\\.";
  int state=0;

  printf("GPU accelerated coulombic potential microbenchmark\n");
  printf("--------------------------------------------------------\n");
  printf("  Single-threaded single-device test run.\n");

  // number of atoms to simulate
  int atomcount = 1000000;
  if(argc==3 && std::strcmp(argv[1],"--atoms")==0) {
    atomcount=std::atoi(argv[2]);
    if(atomcount<=0 || atomcount>1000000) return 1;
  }


  // setup energy grid size
  // XXX this is a large test case to clearly illustrate that even while
  //     the CUDA kernel is running entirely on the GPU, the CUDA runtime
  //     library is soaking up the entire host CPU for some reason.
  volsize.x = 768;
  volsize.y = 768;
  volsize.z = 1;

  // set voxel spacing
  float gridspacing = 0.1;

  // setup CUDA grid and block sizes
  // XXX we have to make a trade-off between the number of threads per
  //     block and the resulting padding size we'll end up with since
  //     each thread will do several consecutive grid cells in this version,
  //     we're using up some of our available parallelism to reduce overhead.
  Bsz.x = BLOCKSIZEX;
  Bsz.y = BLOCKSIZEY;
  Bsz.z = 1;
  Gsz.x = volsize.x / (Bsz.x * UNROLLX);
  Gsz.y = volsize.y / (Bsz.y * UNROLLY);
  Gsz.z = volsize.z;

  // initialize the wall clock timers
  runtimer = wkf_timer_create();
  mastertimer = wkf_timer_create();
  copytimer = wkf_timer_create();
  hostcopytimer = wkf_timer_create();
  copytotal = 0;
  runtotal = 0;
  hostcopytotal = 0;

  printf("Grid size: %d x %d x %d\n", volsize.x, volsize.y, volsize.z);
  printf("Running kernel(atoms:%d, gridspacing %g, z %d)\n", atomcount, gridspacing, 0);

  // allocate and initialize atom coordinates and charges
  if (initatoms(&atoms, atomcount, volsize, gridspacing))
    return -1;

  // allocate and initialize the GPU output array
  int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;
  printf("Allocating %.2fMB of memory for output buffer...\n", volmemsz / (1024.0 * 1024.0));

  iris::Platform platform; check(platform.init(&argc,&argv,true));
  int count=0,device=-1; check(iris_device_count(&count));
  for(int d=0;d<count;++d){int backend=0,type=0;char name[256]={};
    check(iris_device_info(d,iris_backend,&backend,nullptr));
    check(iris_device_info(d,iris_type,&type,nullptr));
    check(iris_device_info(d,iris_name,name,nullptr));
    printf("IRIS device %d: %s backend=%d type=%d\n",d,name,backend,type);
    if(backend==iris_hip && (type&iris_gpu)){device=d;break;}
  }
  if(device<0)throw std::runtime_error("no IRIS HIP GPU");
  energy=static_cast<float*>(calloc(1,volmemsz));
  std::vector<float4> atom_buffer(MAXATOMS);
  float4* datominfo=atom_buffer.data();
  iris::DMem doutput(energy,volmemsz), atom_mem(datominfo,sizeof(float4)*MAXATOMS);
  size_t gws[]={size_t(Gsz.x)*Bsz.x,size_t(Gsz.y)*Bsz.y},lws[]={Bsz.x,Bsz.y};
  printf("starting run...\n");
  wkf_timer_start(mastertimer);

  int iterations=0;
  int atomstart;
  for (atomstart=0; atomstart<atomcount; atomstart+=MAXATOMS) {
    iterations++;
    int runatoms;
    int atomsremaining = atomcount - atomstart;
    if (atomsremaining > MAXATOMS)
      runatoms = MAXATOMS;
    else
      runatoms = atomsremaining;

    printf("%c\r", statestr[state]);
    fflush(stdout);
    state = (state+1) & 3;

    // copy the atoms to the GPU
    wkf_timer_start(copytimer);
    if (copyatoms(atoms + 4*atomstart, runatoms, 0*gridspacing, datominfo))
      return -1;
    wkf_timer_stop(copytimer);
    copytotal += wkf_timer_time(copytimer);

    // RUN the kernel...
    wkf_timer_start(runtimer);
    check(atom_mem.update(datominfo));
    void* args[]={&runatoms,&gridspacing,&doutput,&atom_mem};
    int info[]={sizeof(runatoms),sizeof(gridspacing),iris_rw,iris_r};
    iris::Task task;
    check(task.kernel("cenergy",2,nullptr,gws,lws,4,args,info));
    check(task.submit(device,nullptr,true));
      wkf_timer_stop(runtimer);
    runtotal += wkf_timer_time(runtimer);
  }
  printf("Done\n");

  wkf_timer_stop(mastertimer);
  mastertotal = wkf_timer_time(mastertimer);

  // Copy the GPU output data back to the host and use/store it..
  wkf_timer_start(hostcopytimer);
  iris::Task flush; check(flush.flush_out(doutput)); check(flush.submit(device,nullptr,true));
  wkf_timer_stop(hostcopytimer);
  hostcopytotal=wkf_timer_time(hostcopytimer);

  int i, j;
  for (j=0; j<8; j++) {
    for (i=0; i<8; i++) {
      int addr = j*volsize.x + i;
      printf("[%d] %.1f ", addr, energy[addr]);
    }
    printf("\n");
  }

  printf("Final calculation required %d iterations of %d atoms\n", iterations, MAXATOMS);
  printf("Copy time: %f seconds, %f per iteration\n", copytotal, copytotal / (float) iterations);
  printf("Kernel time: %f seconds, %f per iteration\n", runtotal, runtotal / (float) iterations);
  printf("Total time: %f seconds\n", mastertotal);
  printf("Kernel invocation rate: %f iterations per second\n", iterations / mastertotal);
  printf("GPU to host copy bandwidth: %gMB/sec, %f seconds total\n",
         (volmemsz / (1024.0 * 1024.0)) / hostcopytotal, hostcopytotal);

  double atomevalssec = ((double) volsize.x * volsize.y * volsize.z * atomcount) / (mastertotal * 1000000000.0);
  printf("Efficiency metric, %g billion atom evals per second\n", atomevalssec);

  /* 59/8 FLOPS per atom eval */
  printf("FP performance: %g GFLOPS\n", atomevalssec * (59.0/8.0));

  uint64_t hash=14695981039346656037ULL;
  for(int j=0;j<volmemsz;++j){hash^=reinterpret_cast<unsigned char*>(energy)[j];hash*=1099511628211ULL;}
  printf("OUTPUT_HASH %016llx\n",(unsigned long long)hash);
  free(atoms);
  free(energy);
  check(platform.finalize());

  return 0;
}
