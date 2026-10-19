#!/usr/bin/env python3
"""Auditable statement transcription of the five source GPU bodies, not an algorithm replacement."""
from pathlib import Path
import re,hashlib,json
p=Path(__file__).resolve().parent;source=p.parent/'floydwarshall2-cuda/main.cu';text=source.read_text()
head='''# Source statement transcription: zero-based adapters retain original indexing.
const ws=Int32(32)
const tile=Int32(64)
const ThreadsPerBlock=Int32(1024)
const INT_MAX=typemax(Int32)
struct FZero{A}
    a::A
    offset::Int32
end
FZero(a)=FZero(a,Int32(0))
@inline Base.getindex(a::FZero,i::Integer)=(@inbounds a.a[i+a.offset+Int32(1)])
@inline Base.setindex!(a::FZero,v,i::Integer)=(@inbounds a.a[i+a.offset+Int32(1)]=v)
@inline f_shuffle(v,k)=CUDA.shfl_sync(UInt32(0xffffffff),v,k+Int32(1))
'''
functions=[];bindings=[]
def expr(s):
 s=s.strip().rstrip(';');s=s.replace('g.nodes','nodes').replace('g.nindex','csr_idx').replace('g.nlist','csr_nei').replace('g.eweight','csr_wt')
 s=s.replace('threadIdx.x','tid').replace('blockIdx.x','block').replace('blockDim.x','ThreadsPerBlock')
 s=re.sub(r'__shfl_sync\(~0,\s*([^,]+),\s*([^\)]+)\)',r'f_shuffle(\1, \2)',s)
 s=s.replace('__syncthreads()','CUDA.sync_threads()').replace('/','÷')
 s=re.sub(r'\b(\d+)\b',r'Int32(\1)',s)
 s=re.sub(r'\b(\w+)\+\+',r'\1 += Int32(1)',s)
 return s
for name,args in [('init1','nodes,AdjMat,upper,tid,block'),('init2','nodes,csr_idx,csr_nei,csr_wt,AdjMat,upper,tid,block'),('FW0_64','AdjMat,upper,krows,kcols,tid,block'),('FWrowcol_64','AdjMat,upper,krows,kcols,x,subm1,tid,block'),('FWrem_64','AdjMat,upper,krows,kcols,x,subm1,tid,block')]:
 match=re.search(r'void\s+'+name+r'\s*\(',text);start=text.index('{',match.end());depth=1;end=start+1
 while depth:
  if text[end]=='{':depth+=1
  if text[end]=='}':depth-=1
  end+=1
 body=text[start+1:end-1];lines=['@inline function f_'+name+'!('+args+')'];shared=0;unroll=False
 for raw in body.splitlines():
  line=raw.split('//',1)[0].strip()
  if not line:continue
  if line.startswith('#pragma'):
   assert line=='#pragma unroll 64';unroll=True;continue
  if '__shared__' in line:
   var=re.search(r'mtype\s+(\w+)\[',line)[1]
   if shared==0:lines.append('scratch=CUDA.CuStaticSharedArray(Int32,8192)')
   lines.append(f'{var}=FZero(scratch,Int32({shared*4096}))');shared+=1;continue
  if line=='}':lines.append('end');continue
  if line.startswith('} else {'):lines.append('else');continue
  if line=='else {':
   assert lines[-1]=='end';lines.pop();lines.append('else');continue
  if line.startswith('for ('):
   if unroll:
    assert line=='for (int k = 0; k < tile; k++) {'
    lines.extend(['Base.Cartesian.@nexprs 64 ku -> begin','k=Int32(ku-1)']);unroll=False
   else:
    assert name=='init2' and line=='for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {'
    lines.append('for j in csr_idx[i]:(csr_idx[i+Int32(1)]-Int32(1))')
   continue
  if line.startswith('if ('):
   depth=0;pos=line.index('(')
   for stop in range(pos,len(line)):
    depth+=(line[stop]=='(')-(line[stop]==')')
    if depth==0:break
   cond=line[pos+1:stop];rest=line[stop+1:].strip();lines.append('if '+expr(cond))
   if rest!='{':lines.extend([expr(rest),'end'])
   continue
  line=re.sub(r'^(?:const\s+)?(?:int|mtype)\s+','',line)
  if re.fullmatch(r'\w+\s*,\s*\w+;',line):
   lines.extend(v.strip()+'=Int32(0)' for v in line[:-1].split(','));continue
  assert '{' not in line and '}' not in line,(name,line)
  lines.append(expr(line))
 lines.extend(['nothing','end']);functions.append('\n'.join(lines))
 bindings.append({'source_entrypoint':name,'source_body_sha256':hashlib.sha256(body.encode()).hexdigest(),'julia_helper':'f_'+name+'!','shared_bytes':shared*16384,'source_unroll_requests_preserved':'Base.Cartesian.@nexprs64 static expansion' if '#pragma unroll' in body else 'none'})
(p/'device_math.jl').write_text(head+'\n\n'+'\n\n'.join(functions)+'\n')
(p/'device_transcription.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'entrypoints':bindings,'mapping':'Fivecomplete sourcebodies; Int32 arithmetic/division, same64tile/1024lanes/barriers/1-basedshufflemapping; explicitzero-indexadapter. Two source16KBsharedsymbols use separate4096-word ranges of one8192-word CUDAstaticarray. No foreign GPUdelegation; KA/JACC helper would be hybridCUDAdeviceintrinsics.'},indent=2)+'\n')
