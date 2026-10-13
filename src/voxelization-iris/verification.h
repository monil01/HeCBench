#pragma once
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <vector>
inline void dump_voxels(unsigned n,const unsigned* counts,const unsigned* indices,const uint16_t* features) {
  if(!std::getenv("HECBENCH_VOXEL_DUMP")) return;
  std::vector<unsigned> ids(n);for(unsigned i=0;i<n;++i)ids[i]=i;
  std::sort(ids.begin(),ids.end(),[indices](unsigned a,unsigned b){
    for(int k=0;k<4;++k){if(indices[a*4+k]!=indices[b*4+k])return indices[a*4+k]<indices[b*4+k];}
    return false;
  });
  for(unsigned i:ids)printf("VOXEL %u %u %u %u count=%u half=%04x,%04x,%04x,%04x,%04x\n",
      indices[i*4],indices[i*4+1],indices[i*4+2],indices[i*4+3],counts[i],
      features[i*5],features[i*5+1],features[i*5+2],features[i*5+3],features[i*5+4]);
}
