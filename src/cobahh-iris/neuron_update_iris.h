#include <iris/iris_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

inline void iris_check(int status, const char* operation) {
  if (status != IRIS_SUCCESS) {
    fprintf(stderr, "IRIS %s failed: %d\n", operation, status);
    exit(1);
  }
}

void neurongroup_stateupdater (
    float* __restrict__  _ptr_array_neurongroup_ge,
    float* __restrict__  _ptr_array_neurongroup_gi,
    float* __restrict__  _ptr_array_neurongroup_h,
    float* __restrict__  _ptr_array_neurongroup_m,
    float* __restrict__  _ptr_array_neurongroup_n,
    float* __restrict__  _ptr_array_neurongroup_v,
    float* __restrict__   _ptr_array_neurongroup_lastspike,
    float* __restrict__  _ptr_array_defaultclock_dt,
    float*__restrict__  _ptr_array_defaultclock_t,
    char* __restrict__  _ptr_array_neurongroup_not_refractory,
    const int _N,
    const int iteration ) 
{
  const float dt = _ptr_array_defaultclock_dt[0];
  const float t = _ptr_array_defaultclock_t[0];
  const int    _lio_1 = (int)((0.003f + 1e-3f*dt)/dt);
  const float  _lio_2 = 9.939082f; //1.0f*(0.3291372 * exp(1.0f*(0.055555556 * (-0.063f))/0.001f))/0.001f;
  const float  _lio_3 = -55.555556f; //1.0f*(-0.055555556)/0.001f;
  const float  _lio_4 = 0.00001f; //2980.958 * (0.001f * exp(1.0f*(0.2 * (-0.063f))/0.001f));
  const float  _lio_5 = -200.0f; //1.0f*(-0.2)/0.001f;
  const float  _lio_6 = -0.02016f; //0.32 * (-0.063f);
  const float  _lio_7 = -0.000001f; //- 0.001f * 0.001f;
  const float  _lio_8 = 0; //25.79034 * ((0.001f * 0.001f) * exp(1.0f*(0.25 * (-0.063f))/0.001f));
  const float  _lio_9 = -250.0f; //1.0f*(-0.25)/0.001f;
  const float _lio_10 = 0.00416f; //4.16 * 0.001f;
  const float _lio_11 = 0.02016f; //(-0.32) * (-0.063f);
  const float _lio_12 = -0.01764f; //0.28 * (-0.063f);
  const float _lio_13 = -0.000001f; //(-1.0) * (0.001f * 0.001f);
  const float _lio_14 = 0.000099f; //0.00033546262 * ((0.001f * 0.001f) * exp(1.0f*((-0.2) * (-0.063f))/0.001f));
  const float _lio_15 = 200.0f; //1.0f*0.2/0.001f;
  const float _lio_16 = 0.0112f; //11.2 * 0.001f;
  const float _lio_17 = -0.002016f; //0.032 * (-0.063f);
  const float _lio_18 = 0; //20.085537 * ((0.001f * 0.001f) * exp(1.0f*(0.2 * (-0.063f))/0.001f));
  const float _lio_19 = 0.00048f; //0.48 * 0.001f;
  const float _lio_20 = 0.002016f; //(-0.032) * (-0.063f);
  const float _lio_21 = 132.901474f; //1.0f*(0.6420127 * exp(1.0f*(0.025 * (-0.063f))/0.001f))/0.001f;
  const float _lio_22 = -25.0f ;//1.0f*(-0.025)/0.001f;
  const float _lio_23 = expf(-2000.0f*dt);
  const float _lio_24 = -3.0f; //1.0f*((-0.06f) * 1e-08f)/2e-10f;
  const float _lio_25 = -2700.0f; //1.0f*((-0.09f) * 6e-06f)/2e-10f;
  const float _lio_26 = 5000.0f; //1.0f*(0.05f * 2e-05f)/2e-10f;
  const float _lio_27 = 0; // 1.0f*0.0f/2e-10f;
  const float _lio_28 = -400000000.0f; //1.0f*(-0.08f)/2e-10f;
  const float _lio_29 = -50.0f; //0.0 - (1.0f*1e-08f/2e-10f);
  const float _lio_30 = -30000.0f; //1.0f*(- 6e-06f)/2e-10f;
  const float _lio_31 = 100000.0f; //1.0f*2e-05f/2e-10f;
  const float _lio_32 = 5000000000.0f; //1.0f*1.0/2e-10f;
  const float _lio_33 = expf(-100.0f*dt);

  iris_mem mem_h;
  iris_check(iris_data_mem_create(&mem_h, _ptr_array_neurongroup_h, _N*sizeof(float)), "create dmem");
  iris_mem mem_m;
  iris_check(iris_data_mem_create(&mem_m, _ptr_array_neurongroup_m, _N*sizeof(float)), "create dmem");
  iris_mem mem_n;
  iris_check(iris_data_mem_create(&mem_n, _ptr_array_neurongroup_n, _N*sizeof(float)), "create dmem");
  iris_mem mem_ge;
  iris_check(iris_data_mem_create(&mem_ge, _ptr_array_neurongroup_ge, _N*sizeof(float)), "create dmem");
  iris_mem mem_v;
  iris_check(iris_data_mem_create(&mem_v, _ptr_array_neurongroup_v, _N*sizeof(float)), "create dmem");
  iris_mem mem_gi;
  iris_check(iris_data_mem_create(&mem_gi, _ptr_array_neurongroup_gi, _N*sizeof(float)), "create dmem");
  iris_mem mem_lastspike;
  iris_check(iris_data_mem_create(&mem_lastspike, _ptr_array_neurongroup_lastspike, _N*sizeof(float)), "create dmem");
  iris_mem mem_not_refractory;
  iris_check(iris_data_mem_create(&mem_not_refractory, _ptr_array_neurongroup_not_refractory, _N*sizeof(char)), "create dmem");
  size_t gws[] = {static_cast<size_t>((_N+255)/256)*256};
  size_t lws[] = {256};
  void* params[] = {&mem_h, &mem_m, &mem_n, &mem_ge, &mem_v, &mem_gi, &mem_lastspike, &mem_not_refractory, const_cast<int*>(&_N), const_cast<float*>(&dt), const_cast<float*>(&t), const_cast<int*>(&_lio_1), const_cast<float*>(&_lio_2), const_cast<float*>(&_lio_3), const_cast<float*>(&_lio_4), const_cast<float*>(&_lio_5), const_cast<float*>(&_lio_6), const_cast<float*>(&_lio_7), const_cast<float*>(&_lio_8), const_cast<float*>(&_lio_9), const_cast<float*>(&_lio_10), const_cast<float*>(&_lio_11), const_cast<float*>(&_lio_12), const_cast<float*>(&_lio_13), const_cast<float*>(&_lio_14), const_cast<float*>(&_lio_15), const_cast<float*>(&_lio_16), const_cast<float*>(&_lio_17), const_cast<float*>(&_lio_18), const_cast<float*>(&_lio_19), const_cast<float*>(&_lio_20), const_cast<float*>(&_lio_21), const_cast<float*>(&_lio_22), const_cast<float*>(&_lio_23), const_cast<float*>(&_lio_24), const_cast<float*>(&_lio_25), const_cast<float*>(&_lio_26), const_cast<float*>(&_lio_27), const_cast<float*>(&_lio_28), const_cast<float*>(&_lio_29), const_cast<float*>(&_lio_30), const_cast<float*>(&_lio_31), const_cast<float*>(&_lio_32), const_cast<float*>(&_lio_33)};
  int info[] = {iris_rw, iris_rw, iris_rw, iris_rw, iris_rw, iris_rw, iris_r, iris_w, sizeof(_N), sizeof(dt), sizeof(t), sizeof(_lio_1), sizeof(_lio_2), sizeof(_lio_3), sizeof(_lio_4), sizeof(_lio_5), sizeof(_lio_6), sizeof(_lio_7), sizeof(_lio_8), sizeof(_lio_9), sizeof(_lio_10), sizeof(_lio_11), sizeof(_lio_12), sizeof(_lio_13), sizeof(_lio_14), sizeof(_lio_15), sizeof(_lio_16), sizeof(_lio_17), sizeof(_lio_18), sizeof(_lio_19), sizeof(_lio_20), sizeof(_lio_21), sizeof(_lio_22), sizeof(_lio_23), sizeof(_lio_24), sizeof(_lio_25), sizeof(_lio_26), sizeof(_lio_27), sizeof(_lio_28), sizeof(_lio_29), sizeof(_lio_30), sizeof(_lio_31), sizeof(_lio_32), sizeof(_lio_33)};
  auto start = std::chrono::steady_clock::now();
  for (int rep = 0; rep < iteration; ++rep) {
    iris_task task;
    iris_check(iris_task_create(&task), "create task");
    iris_check(iris_task_kernel(task, "cobahh", 1, nullptr, gws, lws,
                               44, params, info), "kernel");
    if (rep == iteration-1) {
      iris_check(iris_task_dmem_flush_out(task, mem_h), "flush output");
      iris_check(iris_task_dmem_flush_out(task, mem_m), "flush output");
      iris_check(iris_task_dmem_flush_out(task, mem_n), "flush output");
      iris_check(iris_task_dmem_flush_out(task, mem_ge), "flush output");
      iris_check(iris_task_dmem_flush_out(task, mem_v), "flush output");
      iris_check(iris_task_dmem_flush_out(task, mem_gi), "flush output");
      iris_check(iris_task_dmem_flush_out(task, mem_not_refractory), "flush output");
    }
    iris_check(iris_task_submit(task, iris_gpu, nullptr, 1), "submit HIP task");
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  printf("Average kernel execution time %f (us)\n", (time*1e-3f)/iteration);
  iris_check(iris_mem_release(mem_h), "release dmem");
  iris_check(iris_mem_release(mem_m), "release dmem");
  iris_check(iris_mem_release(mem_n), "release dmem");
  iris_check(iris_mem_release(mem_ge), "release dmem");
  iris_check(iris_mem_release(mem_v), "release dmem");
  iris_check(iris_mem_release(mem_gi), "release dmem");
  iris_check(iris_mem_release(mem_lastspike), "release dmem");
  iris_check(iris_mem_release(mem_not_refractory), "release dmem");
}
