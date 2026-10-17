#define main original_hecbench_main
#include "../mlp-cuda/main.cu"
#undef main
#include <dlfcn.h>
int main(){g.seed(123);std::vector<float>v;auto draw=[&](int n){for(int i=0;i<n;i++)v.push_back(distr(g));};draw(128*784);draw(784*512);draw(512*512);draw(512*10);draw(512);draw(512);draw(10);FILE*f=fopen("study_input_f32.bin","wb");if(!f)return 5;fwrite(v.data(),4,v.size(),f);fclose(f);Dl_info info={};dladdr((void*)cublasLtGetVersion,&info);printf("MLP_ABI desc=%zu layout=%zu preference=%zu heuristic=%zu version=%zu library=%s input_values=%zu\n",sizeof(cublasLtMatmulDescOpaque_t),sizeof(cublasLtMatrixLayoutOpaque_t),sizeof(cublasLtMatmulPreferenceOpaque_t),sizeof(cublasLtMatmulHeuristicResult_t),cublasLtGetVersion(),info.dli_fname,v.size());}
