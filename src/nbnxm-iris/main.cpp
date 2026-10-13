#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <hip/hip_runtime.h>
#include "vectypes.h"

typedef float2 Float2;
typedef gmx::BasicVector<float> Float3;
typedef float4 Float4;

#include "constants.h"
#include <iris/iris.hpp>

nbnxn_cj4_t get_cj4(int id) {
  nbnxn_cj4_t value;
  for (int i = 0; i < c_nbnxnGpuJgroupSize; ++i) {
    value.cj[i] = i + id;
  }
  for (int i = 0; i < c_nbnxnGpuClusterpairSplit; ++i) {
    value.imei[i].imask = 0U;
    value.imei[i].excl_ind = 0;
  }
  return value;
}

nbnxn_sci_t get_sci(int id) {
  return {id, 0, 8 * id, 8 * id + 7};
}

nbnxn_excl_t get_excl(int id) {
  nbnxn_excl_t value;
  for (int i = 0; i < c_nbnxnGpuExclSize; ++i) {
    value.pair[i] = 7;
  }
  return value;
}

int main(int argc, char* argv[]) {
  iris::Platform platform;if(platform.init(&argc,&argv,1)!=IRIS_SUCCESS)return 1;
  bool gpu=false;for(int d=0;d<platform.ndevs();d++) {
    int backend=0,type=0;char name[256]={};size_t bytes=0;
    iris_device_info(d,iris_backend,&backend,&bytes);iris_device_info(d,iris_type,&type,&bytes);iris_device_info(d,iris_name,name,&bytes);
    printf("DEVICE backend=%d type=%d name=%s\n",backend,type,name);gpu |= backend==3 && type==256;
  }if(!gpu)return 1;
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const dim3 blocks ( block_x, block_y, 1 );
  const dim3 grids ( 1, 1, grid_z );

  std::vector<Float4> backing_a_xq(NUM_ATOMS);
  Float4* a_xq=backing_a_xq.data();

  std::vector<Float3> backing_a_f(NUM_ATOMS);
  Float3* a_f=backing_a_f.data();

  std::vector<Float3> backing_shiftVec(45);
  Float3* shiftVec=backing_shiftVec.data();

  std::vector<Float3> backing_fShift(45);
  Float3* fShift=backing_fShift.data();

  std::vector<nbnxn_cj4_t> backing_cj4(56881);
  nbnxn_cj4_t* cj4=backing_cj4.data();

  std::vector<nbnxn_sci_t> backing_sci(4806);
  nbnxn_sci_t* sci=backing_sci.data();

  std::vector<nbnxn_excl_t> backing_excl(19205);
  nbnxn_excl_t* excl=backing_excl.data();

  std::vector<int> backing_atomTypes(NUM_ATOMS);
  int* atomTypes=backing_atomTypes.data();

  std::vector<Float2> backing_nbfp(1024);
  Float2* nbfp=backing_nbfp.data();

  for (int i = 0; i < NUM_ATOMS; ++i) {
    a_xq[i] = make_float4(1.0f, 0.5f, 0.25f, 0.125f);
  }
  for (int i = 0; i < NUM_ATOMS; ++i) {
    a_f[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 45; ++i) {
    shiftVec[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 45; ++i) {
    fShift[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 56881; ++i) {
    cj4[i] = get_cj4(i);
  }
  for (int i = 0; i < 4806; ++i) {
    sci[i] = get_sci(i);
  }
  for (int i = 0; i < 19205; ++i) {
    excl[i] = get_excl(i);
  }
  for (int i = 0; i < NUM_ATOMS; ++i) {
    atomTypes[i] = (i % 2);
  }
  for (int i = 0; i < 1024; ++i) {
    nbfp[i] = make_float2(0.5f, 0.25f);
  }

  iris::DMem m_a_xq(a_xq,sizeof(Float4)*(NUM_ATOMS));
  iris::DMem m_a_f(a_f,sizeof(Float3)*(NUM_ATOMS));
  iris::DMem m_shiftVec(shiftVec,sizeof(Float3)*(45));
  iris::DMem m_fShift(fShift,sizeof(Float3)*(45));
  iris::DMem m_cj4(cj4,sizeof(nbnxn_cj4_t)*(56881));
  iris::DMem m_sci(sci,sizeof(nbnxn_sci_t)*(4806));
  iris::DMem m_excl(excl,sizeof(nbnxn_excl_t)*(19205));
  iris::DMem m_atomTypes(atomTypes,sizeof(int)*(NUM_ATOMS));
  iris::DMem m_nbfp(nbfp,sizeof(Float2)*(1024));
  auto dump=[&]() {
    if(const char* path=getenv("HECBENCH_ARTIFACT")) {
      FILE* out=fopen(path,"ab");if(!out)exit(3);
      fwrite(a_f,sizeof(Float3),NUM_ATOMS,out);fwrite(fShift,sizeof(Float3),45,out);fwrite(cj4,sizeof(nbnxn_cj4_t),56881,out);fclose(out);
    }
  };
  if(const char* path=getenv("HECBENCH_ARTIFACT")) { FILE* out=fopen(path,"wb");if(!out)exit(3);fclose(out); }
  // Warming-up
  {
    int arg0=32; float arg1=1; float arg2=3.12341; float arg3=138.935; bool arg4=0;
    void* args[]={&m_a_xq,&m_a_f,&m_shiftVec,&m_fShift,&m_cj4,&m_sci,&m_excl,&m_atomTypes,&m_nbfp,&arg0,&arg1,&arg2,&arg3,&arg4};
    int info[]={iris_r,iris_rw,iris_r,iris_rw,iris_rw,iris_r,iris_r,iris_r,iris_r,sizeof(arg0),sizeof(arg1),sizeof(arg2),sizeof(arg3),sizeof(arg4)};
    size_t gws[]={grids.x*blocks.x,grids.y*blocks.y,grids.z*blocks.z},lws[]={blocks.x,blocks.y,blocks.z};
    iris::Task task;if(task.kernel("nbnxmKernelTest",3,nullptr,gws,lws,14,args,info)!=IRIS_SUCCESS)return 2;
    task.flush_out(m_a_f);task.flush_out(m_fShift);task.flush_out(m_cj4);
    if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
  }


  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; ++i) {
    {
    int arg0=32; float arg1=1; float arg2=3.12341; float arg3=138.935; bool arg4=0;
    void* args[]={&m_a_xq,&m_a_f,&m_shiftVec,&m_fShift,&m_cj4,&m_sci,&m_excl,&m_atomTypes,&m_nbfp,&arg0,&arg1,&arg2,&arg3,&arg4};
    int info[]={iris_r,iris_rw,iris_r,iris_rw,iris_rw,iris_r,iris_r,iris_r,iris_r,sizeof(arg0),sizeof(arg1),sizeof(arg2),sizeof(arg3),sizeof(arg4)};
    size_t gws[]={grids.x*blocks.x,grids.y*blocks.y,grids.z*blocks.z},lws[]={blocks.x,blocks.y,blocks.z};
    iris::Task task;if(task.kernel("nbnxmKernelTest",3,nullptr,gws,lws,14,args,info)!=IRIS_SUCCESS)return 2;
    task.flush_out(m_a_f);task.flush_out(m_fShift);task.flush_out(m_cj4);
    if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
  }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (w/o shift): %f (us)\n", (time * 1e-3f) / repeat);
  dump();

#ifdef DEBUG
  float f0 = 0, f1 = 0, f2 = 0; 
  for (int i = 0; i < NUM_ATOMS; ++i) {
    f0 += a_f[i][0];
    f1 += a_f[i][1];
    f2 += a_f[i][2];
  }
  printf("Checksum (a_f): %f %f %f\n", f0, f1, f2);

  f0 = 0, f1 = 0, f2 = 0; 
  for (int i = 0; i < 45; ++i) {
    f0 += fShift[i][0];
    f1 += fShift[i][1];
    f2 += fShift[i][2];
  }
  printf("Checksum (fShift): %f %f %f\n", f0, f1, f2);
#endif

  for (int i = 0; i < NUM_ATOMS; ++i) {
    a_xq[i] = make_float4(1.0f, 0.5f, 0.25f, 0.125f);
  }
  for (int i = 0; i < NUM_ATOMS; ++i) {
    a_f[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 45; ++i) {
    shiftVec[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 45; ++i) {
    fShift[i] = Float3(1.0f, 0.5f, 0.25f);
  }
  for (int i = 0; i < 56881; ++i) {
    cj4[i] = get_cj4(i);
  }
  for (int i = 0; i < 4806; ++i) {
    sci[i] = get_sci(i);
  }
  for (int i = 0; i < 19205; ++i) {
    excl[i] = get_excl(i);
  }
  for (int i = 0; i < NUM_ATOMS; ++i) {
    atomTypes[i] = (i % 2);
  }
  for (int i = 0; i < 1024; ++i) {
    nbfp[i] = make_float2(0.5f, 0.25f);
  }

  m_a_xq.update(a_xq);
  m_a_f.update(a_f);
  m_shiftVec.update(shiftVec);
  m_fShift.update(fShift);
  m_cj4.update(cj4);
  m_sci.update(sci);
  m_excl.update(excl);
  m_atomTypes.update(atomTypes);
  m_nbfp.update(nbfp);
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; ++i) {
    {
    int arg0=32; float arg1=1; float arg2=3.12341; float arg3=138.935; bool arg4=1;
    void* args[]={&m_a_xq,&m_a_f,&m_shiftVec,&m_fShift,&m_cj4,&m_sci,&m_excl,&m_atomTypes,&m_nbfp,&arg0,&arg1,&arg2,&arg3,&arg4};
    int info[]={iris_r,iris_rw,iris_r,iris_rw,iris_rw,iris_r,iris_r,iris_r,iris_r,sizeof(arg0),sizeof(arg1),sizeof(arg2),sizeof(arg3),sizeof(arg4)};
    size_t gws[]={grids.x*blocks.x,grids.y*blocks.y,grids.z*blocks.z},lws[]={blocks.x,blocks.y,blocks.z};
    iris::Task task;if(task.kernel("nbnxmKernelTest",3,nullptr,gws,lws,14,args,info)!=IRIS_SUCCESS)return 2;
    task.flush_out(m_a_f);task.flush_out(m_fShift);task.flush_out(m_cj4);
    if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)return 2;
  }  // compute shift
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (w/ shift): %f (us)\n", (time * 1e-3f) / repeat);
  dump();

#ifdef DEBUG
  f0 = 0, f1 = 0, f2 = 0; 
  for (int i = 0; i < NUM_ATOMS; ++i) {
    f0 += a_f[i][0];
    f1 += a_f[i][1];
    f2 += a_f[i][2];
  }
  printf("Checksum (a_f): %f %f %f\n", f0, f1, f2);

  f0 = 0, f1 = 0, f2 = 0; 
  for (int i = 0; i < 45; ++i) {
    f0 += fShift[i][0];
    f1 += fShift[i][1];
    f2 += fShift[i][2];
  }
  printf("Checksum (fShift): %f %f %f\n", f0, f1, f2);
#endif










  int errors=platform.error_count();platform.finalize();return errors ? 1 : 0;
}
