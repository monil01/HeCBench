#!/usr/bin/env python3
"""Explicit CUDA12 API qualification only: legacy texture references -> linear texture objects."""
from pathlib import Path
import re,argparse,hashlib,json,difflib
p=Path(__file__).resolve();R=p.parents[3];a=argparse.ArgumentParser();a.add_argument('--source-copy',type=Path,required=True);x=a.parse_args();original=R/'src/miniDGS-cuda/src/MaxwellsKernel3d.cu';before=original.read_text();s=before;types={name:typ for typ,name in re.findall(r'texture<(float4|float), 1, cudaReadModeElementType> (t_\w+);',s)};assert len(types)==10
s=s.replace('#include <cuda.h>','#include <cuda.h>\n#include <cuda_runtime.h>')
s=re.sub(r'texture<(float4|float), 1, cudaReadModeElementType> (t_\w+);',r'__device__ cudaTextureObject_t \2;',s)
helper='''\n// ModernAPIcompatibility: equivalent element-mode linear texture fetches.
template<class T>static cudaError_t study_texture_create(void*pointer,size_t bytes,cudaTextureObject_t*handle){
 cudaResourceDesc resource={};resource.resType=cudaResourceTypeLinear;
 resource.res.linear.devPtr=pointer;resource.res.linear.desc=cudaCreateChannelDesc<T>();resource.res.linear.sizeInBytes=bytes;
 cudaTextureDesc texture={};texture.readMode=cudaReadModeElementType;
 return cudaCreateTextureObject(handle,&resource,&texture,nullptr);
}
'''
s=s.replace('static float *c_LIFT;',helper+'\nstatic float *c_LIFT;')
binds=[]
def bind(m):
 name,pointer,size=m.groups();typ=types[name];binds.append(name)
 return '{ cudaTextureObject_t handle=0; checkError(study_texture_create<'+typ+'>('+pointer+','+size+',&handle)); checkError(cudaMemcpyToSymbol('+name+',&handle,sizeof(handle))); }'
s=re.sub(r'checkError\(cudaBindTexture\(0,\s*(t_\w+),\s*(\w+),\s*(\w+)\)\);',bind,s);assert len(binds)==6,binds
for name,typ in types.items():s=re.sub(r'tex1Dfetch\('+name+r',',f'tex1Dfetch<{typ}>({name},',s)
assert 'cudaBindTexture'not in s and'texture<'not in s
out=x.source_copy/'src/MaxwellsKernel3d.cu';out.write_text(s);diff=''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='immutableoriginal',tofile='ownedCUDA12textureobjectcopy'));(p.parent/'texture_object_migration.patch').write_text(diff)
(p.parent/'migration_provenance.json').write_text(json.dumps(dict(original=str(original.relative_to(R)),original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),qualified_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),patch_sha256=hashlib.sha256(diff.encode()).hexdigest(),texture_bindings=binds,qualification='LegacytextureAPI→CUDA12lineartextureobjects; samechanneltypes/byteextents/elementfetchindices andallkernelarithmeticunchanged. Texturehandlecreation/API/globalhandleloads differ; notuntouchednativeexecution, no mathematicalscientificrepair.',native_integration_adapted=True,exclude_untouched_native_aggregates=True),indent=2)+'\n')
print('qualified',out)
