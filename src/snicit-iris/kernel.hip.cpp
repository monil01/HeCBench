#include <hip/hip_runtime.h>
#include <cmath>


namespace SNICIT_BEY{
extern "C" __global__ void dense_input(
  const float* Y0,
  const float* weight,
  const float* bias,
  const int M, const int N, const int K,
  float* Y1
) {
    __shared__ float shRow[512];
    if (threadIdx.y == 0) {
        shRow[threadIdx.x] = bias[threadIdx.x];
    }

    __syncthreads();
    // if (threadIdx.x+threadIdx.y == 0 && blockIdx.x == 0) {
    //     printf("blockDim.x%d blockDim.y%d", blockDim.x, blockDim.y);
    // }
    for (int i = threadIdx.y; i < N; i += blockDim.y) {
        float valY = Y0[blockIdx.x * N + i];
        if(valY == 0) {
            continue;
        }
        float valW = weight[i*K+threadIdx.x];
        atomicAdd(&shRow[threadIdx.x], valY * valW);
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        Y1[blockIdx.x * K+threadIdx.x] = min(float(1.0), max(float(0), shRow[threadIdx.x]));
    }
}


extern "C" __global__ void sparse_hidden(

  const float* Y0,
  const int* roffW,
  const int* colsW,
  const float* valsW,
  const float* bias,
  const int M, const int N, const int K,
  float* Y1
) {
    // (8, 128)
    __shared__ float shRow[512];
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    if (tid < K) {
        shRow[tid] = bias[tid];
    }
    __syncthreads();

    for (int i = threadIdx.y; i < N; i += blockDim.y) {
        float valY = Y0[blockIdx.x * N + i];
        if(valY == 0) {
            continue;
        }

        int begOffW = roffW[i] + threadIdx.x;
        int endOffW = roffW[i + 1];
        for(int k = begOffW; k < endOffW; k += blockDim.x) { // += blockDim.x
            int colW = colsW[k];
            float valW = valsW[k];
            atomicAdd(&shRow[colW], valY * valW);
        }
    }
    __syncthreads();
    if (tid < K) {
        Y1[blockIdx.x * K+tid] = min(float(1.0), max(float(0), shRow[tid]));
    }
}

extern "C" __global__ void dense_output(
  const float* Y0,
  const float* weight,
  const float* bias,
  const int M, const int N, const int K,
  float* Y1
) {
    __shared__ float shRow[512];
    if (threadIdx.y == 0) {
        shRow[threadIdx.x] = bias[threadIdx.x];
    }

    __syncthreads();
    // if (threadIdx.x+threadIdx.y == 0 && blockIdx.x == 0) {
    //     printf("blockDim.x%d blockDim.y%d", blockDim.x, blockDim.y);
    // }
    for (int i = threadIdx.y; i < N; i += blockDim.y) {
        float valY = Y0[blockIdx.x * N + i];
        if(valY == 0) {
            continue;
        }
        float valW = weight[i*K+threadIdx.x];
        atomicAdd(&shRow[threadIdx.x], valY * valW);
    }
    __syncthreads();
    if (threadIdx.y == 0) {
        Y1[blockIdx.x * K+threadIdx.x] = shRow[threadIdx.x];
    }
}

extern "C" __global__ void check_acc(
    float* Y, int num_classes, int num_input, int* label, int *cnt
) {
    __shared__ int shcnt[1];
    if (threadIdx.x == 0)
        shcnt[0] = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < num_input; i += blockDim.x) {
        int argmax = 0;
        float tmpmax = -10000.0;
        for (int j = 0; j < num_classes; j++) {
            if (Y[i*num_classes+j] > tmpmax) {
                argmax = j;
                tmpmax = Y[i*num_classes+j];
            }
        }
        // printf("label[i]=%d argmax=%d, tmpmax = %f\n", label[i], argmax, tmpmax);
        if (argmax == label[i])
            atomicAdd(&shcnt[0], 1);
    }
    // printf("shcnt[0]=%d", shcnt[0]);

    __syncthreads();
    if (threadIdx.x == 0)
        cnt[0] = shcnt[0];
}

}



namespace SNICIT_BEY{

extern "C" __global__ void y_star_gen(
    const float* Y0,
    int *y_star_row,
    const int num_input,
    const int neurons,
    const int seed_size
) {
    int row_idx = threadIdx.y * num_input / blockDim.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ float shRow[512]; // combined diff_arr and tmp_star_row here
    if (threadIdx.x == 0) {
        shRow[neurons+seed_size+threadIdx.y] = (float)row_idx;
    }
    __syncthreads();
    for (int i = 0; i < seed_size; i++) {
        if (shRow[neurons+seed_size+i]!=-1.0) {
            if (tid < neurons) {
                shRow[tid] = Y0[neurons*(int)shRow[neurons+seed_size+i]+tid]; // to be compared
            }
            if (tid < seed_size) {
                shRow[neurons+tid] = 0;
            }
            __syncthreads();
            if (shRow[neurons+seed_size+threadIdx.y] != -1.f) {
                for (int j = threadIdx.x; j < neurons; j += blockDim.x) {
                    if (abs(Y0[neurons*row_idx+j] - shRow[j]) > 0.03f) {
                        atomicAdd(&shRow[neurons+threadIdx.y], 1);
                    }
                }
            }
            __syncthreads();
            if (threadIdx.y!=i && shRow[neurons+threadIdx.y] < neurons*0.03f) {
                shRow[neurons+seed_size+threadIdx.y] = -1.f;
            }
            __syncthreads();
        }
    }
    if (tid < seed_size) {
        y_star_row[tid] = (int)shRow[neurons+seed_size+tid];
    }
    __syncthreads();
}

extern "C" __global__ void coarse_cluster(
    float* Y0,
    const int *y_star_row,
    bool *ne_record,
    const int y_star_cnt,
    int *centroid_LUT,
    const int neurons
) {
    if (centroid_LUT[blockIdx.x] == -1) {
        ne_record[blockIdx.x] = true;
        return;
    }
    __shared__ float thisRow[512];
    // __shared__ float diff_arr[60]; // estimated max y* num
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    if (tid < neurons) {
        thisRow[tid] = Y0[blockIdx.x*neurons+tid];
    }
    if (tid < y_star_cnt) {
        thisRow[neurons+tid] = 0;
    }
    __syncthreads();
    for(int i = threadIdx.x; i < neurons; i += blockDim.x) {
        if (abs(Y0[neurons*y_star_row[threadIdx.y]+i] - thisRow[i]) > 0.04f) {
            atomicAdd(&thisRow[neurons+threadIdx.y], 1);
        }
    }
    __syncthreads();
    int argmin=-10;
    float min_num = neurons+1;
    if (tid == 0) {
        for (int i = 0; i < y_star_cnt; i++) {
            if (min_num > thisRow[neurons+i]) {
                min_num = thisRow[neurons+i];
                argmin = y_star_row[i];
            }
        }
        centroid_LUT[blockIdx.x] = argmin;

    }
    __syncthreads();
    argmin = centroid_LUT[blockIdx.x];
    float v = ((tid < neurons) && (abs(thisRow[tid]-Y0[neurons*argmin+tid])>0.04f)) ?
        thisRow[tid]-Y0[neurons*argmin+tid] : 0;
    if (tid < neurons) {
        Y0[blockIdx.x*neurons+tid] = v; // change blockIdx.x to argmin
    }
    int count = __syncthreads_count(v > 0);
    if (tid == 0) {
        if (count == 0) ne_record[blockIdx.x] = false;
        else ne_record[blockIdx.x] = true;
    }

}

extern "C" __global__ void sparse_hidden_post(
    const int *rowsY,
    const float* Y0,
    const int* roffW,
    const int* colsW,
    const float* valsW,
    const int M, const int N, const int K,
    float* Y1
) {
    // (8, 128)
    __shared__ float shRow[512];
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    int rid = rowsY[blockIdx.x];
    if (tid < K) {
        shRow[tid] = 0;
    }
    __syncthreads();

    for (int i = threadIdx.y; i < N; i += blockDim.y) {
        float valY = Y0[rid * N + i];
        if(valY == 0) {
            continue;
        }

        int begOffW = roffW[i] + threadIdx.x;
        int endOffW = roffW[i + 1];
        for(int k = begOffW; k < endOffW; k += blockDim.x) { // += blockDim.x
            int colW = colsW[k];
            float valW = valsW[k];
            atomicAdd(&shRow[colW], valY * valW);
        }
    }
    __syncthreads();
    if (tid < K) {
        Y1[rid * K+tid] = shRow[tid];
    }
}

extern "C" __global__ void update_post(
    const int *rowsY,
    const int *centroid_LUT,
    const float* Y0,
    const float* bias,
    const int neurons,
    bool* ne_record,
    float* Y1
) {
    int tid = threadIdx.x;
    int rid = rowsY[blockIdx.x];
    float b = bias[threadIdx.x];
    if (centroid_LUT[rid] == -1) {
        Y1[rid * neurons+tid] = min(float(1.0), max(float(0), Y0[rid * neurons+tid]+b));//Y0[rid * neurons+tid];
        ne_record[rid] = true;
        return;
    }
    float wy_centroid = Y0[neurons * centroid_LUT[rid] + tid];
    float wdelta_y = Y0[neurons * rid + tid];
    float true_diff = min(float(1.0), max(float(0), wy_centroid+b+wdelta_y))-min(float(1.0), max(float(0), wy_centroid+b));
    float val = (abs(true_diff)>0.05)?true_diff:0;
    int count = __syncthreads_count(val != 0);
    Y1[rid * neurons+tid] = val;
    if (tid == 0) {
        if (count == 0) ne_record[rid] = false;
        else ne_record[rid] = true;
    }
}

extern "C" __global__ void recover(
    float* Y0,
    const int *centroid_LUT,
    const int neurons
) {
    __shared__ float shRow[512];
    if (centroid_LUT[blockIdx.x] == -1) {
        return;
    }
    int tid = threadIdx.x;
    shRow[tid] = Y0[blockIdx.x*neurons+tid] + Y0[centroid_LUT[blockIdx.x]*neurons+tid];
    __syncthreads();
    Y0[blockIdx.x*neurons+tid] = shRow[tid];
}

}
