import sys,pathlib,re
sys.path.insert(0,'/tmp/logic_rewrite_clang_bindings')
from clang import cindex
cindex.Config.set_library_file('/lib/x86_64-linux-gnu/libclang-18.so.18')
root=pathlib.Path('/home/imo/HeCBench_julia/src/logic-rewrite-cuda')
s=(root/'rewrite.cu').read_text();s=re.sub(r'<<<.*?>>>','',s,flags=re.S)
pre='''#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __managed__
#define __shared__
#include <cuda_runtime.h>
struct __study_dim {unsigned x,y,z;};
extern __study_dim blockIdx,blockDim,threadIdx,gridDim;
void __syncthreads();
template<class T,class U>T atomicAdd(T*,U);
template<class T,class U>T atomicExch(T*,U);
template<class T,class U>T atomicSub(T*,U);
template<class T,class U,class V>T atomicCAS(T*,U,V);
'''
t=cindex.Index.create().parse(str(root/'ast_probe.cpp'),args=['-x','c++','-std=c++17','-I'+str(root),'-I/usr/include/c++/12','-I/usr/include/x86_64-linux-gnu/c++/12','-isystem','/usr/lib/gcc/x86_64-linux-gnu/12/include','-ferror-limit=0'],unsaved_files=[(str(root/'ast_probe.cpp'),pre+s)])
for d in list(t.diagnostics)[:25]:print(d)
print('diags',len(list(t.diagnostics)))
for c in t.cursor.get_children():
 if c.location.file and c.location.file.name.endswith('ast_probe.cpp') and c.is_definition():print(c.kind,c.spelling)
