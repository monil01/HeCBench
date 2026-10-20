from pathlib import Path
import re,json,hashlib
P=Path(__file__).resolve().parent;s=(P/'native_diagnostics/qualified_preinstrumentation.cu.txt').read_text();names=['MaxwellsGPU_VOL_Kernel3D','MaxwellsGPU_SURF_Kernel3D','MaxwellsGPU_RK_Kernel3D','partial_get_kernel3d'];chunks=[]
for name in names:
 start=s.index('__global__ void '+name);b=s.index('{',start);depth=1;j=b+1
 while depth:
  depth+=(s[j]=='{')-(s[j]=='}');j+=1
 chunks.append(s[start:j]+'\n')
head='#include <cuda_runtime.h>\n#define p_Np 84\n#define p_Nfp 28\n#define p_Nfaces 4\n#define p_Nfields 6\n#define BSIZE 96\n'+''.join('__device__ cudaTextureObject_t '+n+';\n'for n in ['t_Q','t_partQ','t_vgeo','t_DrDsDt','t_LIFT','t_surfinfo'])
(P/'source_kernels.cuh').write_text(head+'\n'.join(chunks));(P/'native_kernel_extraction.json').write_text(json.dumps(dict(source='src/miniDGS-julia/native_diagnostics/qualified_preinstrumentation.cu.txt',source_sha256=hashlib.sha256(s.encode()).hexdigest(),kernel_names=names,body_hashes={n:hashlib.sha256(c.encode()).hexdigest()for n,c in zip(names,chunks)},header_sha256=hashlib.sha256((P/'source_kernels.cuh').read_bytes()).hexdigest(),arithmetic_and_indexing_unchanged_from_qualified_source=True,legacy_API_migration_separate='native_compatibility/migration_provenance.json'),indent=2)+'\n')
