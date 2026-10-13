#include <hip/hip_runtime.h>
// Each RHS is independent. Sequential rows preserve triangular dependencies.
extern "C" __global__ void csr_forward_solve(const int* offsets,const int* columns,
                                             const float* values,const float* b,float* c,
                                             int m,int n,float alpha) {
    int rhs=blockIdx.x*blockDim.x+threadIdx.x;
    if(rhs>=n) return;
    for(int row=0;row<m;++row) {
        float sum=alpha*b[row*n+rhs], diagonal=0.0f;
        for(int p=offsets[row];p<offsets[row+1];++p) {
            int col=columns[p];
            if(col<row) sum-=values[p]*c[col*n+rhs];
            else if(col==row) diagonal=values[p];
        }
        c[row*n+rhs]=sum/diagonal;
    }
}
