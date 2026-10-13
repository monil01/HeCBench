#include <cuda_runtime.h>
#include "nearestNeighbor.h"
extern "C" __global__ void 
nn (const int numRecords, const float lat, const float lng,
    const LatLong *__restrict__ locations,
    float*__restrict__ distances) 
{
  int gid = blockDim.x * blockIdx.x + threadIdx.x;
  if (gid < numRecords) {
    LatLong latLong = locations[gid];
    distances[gid] = sqrtf((lat-latLong.lat)*(lat-latLong.lat)+
        (lng-latLong.lng)*(lng-latLong.lng));
  }
}
