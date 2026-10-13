#include <hip/hip_runtime.h>
#include <float.h>
// copy the feature to a feature swap region
extern "C" __global__
void feature_transpose (float* feature_swap,
                        const float* feature,
                        const int nfeatures,
                        const int npoints)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 
  if (tid < npoints) {
    for(int i = 0; i <  nfeatures; i++)
      feature_swap[i * npoints + tid] = feature[tid * nfeatures + i];
  }
}

extern "C" __global__
void find_membership (const float*__restrict__ feature,
                      const float*__restrict__ cluster,
                              int*__restrict__ member, 
                      const int nclusters,
                      const int nfeatures,
                      const int npoints)
{
  int point_id = blockIdx.x * blockDim.x + threadIdx.x; 
  if (point_id < npoints) {
    int index = 0;
    float min_dist = FLT_MAX;
    for (int i = 0; i < nclusters; i++) {
      float dist = 0;
      float ans  = 0;
      for (int l = 0; l < nfeatures; l++) {
        ans += (feature[l * npoints + point_id] - cluster[i * nfeatures + l]) * 
               (feature[l * npoints + point_id] - cluster[i * nfeatures + l]) ; 
      }
      dist = ans;
      if (dist < min_dist) {
        min_dist = dist;
        index    = i;
      }
    }
    member[point_id] = index;
  }
}
