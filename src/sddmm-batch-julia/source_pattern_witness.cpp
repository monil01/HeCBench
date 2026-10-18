// Original utility calls: RNG reseeding repeats structural sparsity per batch.
#include <cstdlib>
#include <cstdio>
#include <vector>
#include "../sddmm-batch-cuda/utils.h"
int main(){int m=17,n=19,nz=37;std::vector<int>firstoff,firstcols;int differences=0,value_differences=0;std::vector<float>firstvalues;
for(int b=0;b<4;++b){std::vector<float>C(m*n),values(nz);std::vector<int>off(m+1),cols(nz);init_matrix(C.data(),m,n,nz);init_csr(off.data(),values.data(),cols.data(),C.data(),m,n,nz);if(b==0){firstoff=off;firstcols=cols;firstvalues=values;}else{differences+=off!=firstoff||cols!=firstcols;value_differences+=values!=firstvalues;}}
printf("{\"batches\":4,\"structure_differing_from_batch0\":%d,\"values_differing_from_batch0\":%d,\"source_srand123_reseed_per_init_matrix\":true}\n",differences,value_differences);return differences?1:0;}
