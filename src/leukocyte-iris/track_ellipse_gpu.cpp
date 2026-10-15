#include <chrono>
#include "iris_helpers.h"
#include <vector>
#include "track_ellipse.h"

// Host and device arrays to hold matrices for all cells
// (so we can copy to and from the device in a single transfer)

// The number of work items per work group
#define LOCAL_WORK_SIZE 256
#define FP_TYPE float
#define FP_CONST(num) num##f
#define PI_FP32 FP_CONST(3.14159)
#define ONE_OVER_PI (FP_CONST(1.0) / PI_FP32)
#define MU FP_CONST(0.5)
#define LAMBDA (FP_CONST(8.0) * MU + FP_CONST(1.0))
#define NEXT_LOWEST_POWER_OF_TWO 256



// Host function that launches a GPU kernel to compute the MGVF matrices for the specified cells
void IMGVF_GPU(MAT **IE, MAT **IMGVF,
    double vx, double vy, double e, int max_iterations, double cutoff, int num_cells) {

  // Initialize the data on the GPU
  // Allocate array of offsets to each cell's image
  size_t mem_size = sizeof(int) * num_cells;
  int* host_I_offsets = (int *) malloc(mem_size);

  // Allocate arrays to hold the dimensions of each cell's image
  int* host_m_array = (int *) malloc(mem_size);
  int* host_n_array = (int *) malloc(mem_size);

  // Figure out the size of all of the matrices combined
  int i, j;
  size_t total_size = 0;
  for (int cell_num = 0; cell_num < num_cells; cell_num++) {
    MAT *I = IE[cell_num];
    size_t size = I->m * I->n;
    total_size += size;
  }
  size_t total_mem_size = total_size * sizeof(float);

  // Allocate host memory just once for all cells
  float* host_I_all = (float *) malloc(total_mem_size);

  // Copy each initial matrix into the allocated host memory
  int offset = 0;
  for (int cell_num = 0; cell_num < num_cells; cell_num++) {
    MAT *I = IE[cell_num];

    // Determine the size of the matrix
    int m = I->m, n = I->n;
    int size = m * n;

    // Store memory dimensions
    host_m_array[cell_num] = m;
    host_n_array[cell_num] = n;

    // Store offsets to this cell's image
    host_I_offsets[cell_num] = offset;

    // Copy matrix I (which is also the initial IMGVF matrix) into the overall array
    for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
        host_I_all[offset + (i * n) + j] = (float) m_get_val(I, i, j);

    offset += size;
  }

  // Convert double-precision parameters to single-precision
  float vx_float = (float) vx;
  float vy_float = (float) vy;
  float e_float = (float) e;
  float cutoff_float = (float) cutoff;

  if (num_cells == 0) {
    free(host_m_array); free(host_n_array); free(host_I_all); free(host_I_offsets);
    return;
  }
  std::vector<float> result(host_I_all,host_I_all+total_size);
  {
  iris::DMem moff(host_I_offsets,sizeof(int)*num_cells);
  iris::DMem mm(host_m_array,sizeof(int)*num_cells), mn(host_n_array,sizeof(int)*num_cells);
  iris::DMem mi(host_I_all,total_mem_size), mout(result.data(),total_mem_size);
  auto start = std::chrono::steady_clock::now();
  {
    iris::Task task;
    void* params[]={&mout,&mi,&moff,&mm,&mn,&vx_float,&vy_float,&e_float,&cutoff_float,&max_iterations};
    int info[]={iris_rw,iris_r,iris_r,iris_r,iris_r,sizeof(vx_float),sizeof(vy_float),sizeof(e_float),sizeof(cutoff_float),sizeof(max_iterations)};
    size_t global[]={static_cast<size_t>(num_cells)*LOCAL_WORK_SIZE}, local[]={LOCAL_WORK_SIZE};
    iris_check(task.kernel("kernel_IMGVF",1,nullptr,global,local,10,params,info));
    iris_check(task.flush_out(mout)); iris_check(task.submit(iris_gpu,nullptr,1));
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count();
  printf("Kernel execution time (IMGVF): %f (s)\n",time*1e-9f);
  static int diagnostic_call=0;
  leuk_mgvf(diagnostic_call++,num_cells,int(total_size),host_I_offsets,host_m_array,host_n_array,host_I_all,result.data());
  memcpy(host_I_all,result.data(),total_mem_size);
  } // All DMem bindings die before the host allocations are released.

  // Copy each result matrix into its appropriate host matrix
  offset = 0;
  for (int cell_num = 0; cell_num < num_cells; cell_num++) {
    MAT *IMGVF_out = IMGVF[cell_num];

    // Determine the size of the matrix
    int m = IMGVF_out->m, n = IMGVF_out->n, i, j;
    // Pack the result into the matrix
    for (i = 0; i < m; i++)
      for (j = 0; j < n; j++) {
#ifdef DEBUG
        printf("host_IMGVF: %f\n",host_I_all[offset + (i * n) + j]);
#endif

        m_set_val(IMGVF_out, i, j, (double) host_I_all[offset + (i * n) + j]);
      }

    offset += (m * n);
  }

  // Free host memory
  free(host_m_array);
  free(host_n_array);
  free(host_I_all);
  free(host_I_offsets);
}
