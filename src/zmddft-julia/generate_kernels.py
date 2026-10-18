"""Mechanical straight-line Spiral C→Julia transcription; no algorithm substitution."""
from pathlib import Path
import re,json,hashlib,argparse
P=Path(__file__).resolve().parent;R=P.parents[1];source=R/'src/zmddft-cuda/main.cu';s=source.read_text();parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['cuda_jl','ka','jacc'],required=True);args=parser.parse_args()
if args.model=='jacc':
 reference=P/'zmddft-julia-cuda.jl/main.jl';out=reference.read_text().split('function factory(')[0]
 out=out.replace('using CUDA\n','using CUDA,JACC\nJACC.@init_backend\n').replace('function stage0(','function stage0(i,').replace('function stage1(','function stage1(i,').replace('function stage2(','function stage2(i,').replace('Int32(threadIdx().x-1)','Int32((i-1)%64)').replace('Int32(blockIdx().x-1)','Int32((i-1)÷64)')
 out+='function makearray(a)\n d=JACC.array(a);d isa CUDA.CuArray{Float64}||error("JACCexpectedCUDAF64array");d\nend\nfunction factory(X,P1,P2,Y,D3)\n spec=JACC.launch_spec(;threads=64,blocks=16384,shmem_size=0,sync=false)\n function launch()\n  JACC.parallel_for(spec,1048576,stage0,X,P1,D3)\n  JACC.parallel_for(spec,1048576,stage1,P1,P2,D3)\n  JACC.parallel_for(spec,1048576,stage2,P2,Y,D3)\n  nothing\n end\n launch\nend\ninclude("../study_common.jl")\nzmddft_study("jacc",makearray,factory)\n'
 dest=P/'zmddft-julia-JACC';dest.mkdir(exist_ok=True);f=dest/'main.jl';f.write_text(out)
 (P/'kernel_generation_jacc.json').write_text(json.dumps(dict(source=str(source.relative_to(R)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reference_Julia_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),output=str(f.relative_to(R)),output_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),generated_Julia_bytes=len(out.encode()),authored_device_definitions=3,source_assignment_body_reused=True,shared_reference_generation_accounted_once=True,foreign_device_extensions=['CUDA.CuStaticSharedArray','CUDA.sync_threads']),indent=2)+'\n')
 print('GeneratedJACC',len(out.encode()));raise SystemExit
if args.model=='ka':
 reference=P/'zmddft-julia-cuda.jl/main.jl';out=reference.read_text().split('function factory(')[0]
 out=out.replace('using CUDA\n','using CUDA,KernelAbstractions\n').replace('function stage','@kernel function stage').replace('Int32(threadIdx().x-1)','Int32(@index(Local,Linear)-1)').replace('Int32(blockIdx().x-1)','Int32(@index(Group,Linear)-1)').replace('CUDA.CuStaticSharedArray(Float64,2048)','@localmem Float64 (2048,)').replace('CUDA.sync_threads()','@synchronize')
 out+='function makearray(a)\n d=KernelAbstractions.allocate(CUDABackend(),Float64,(length(a),));copyto!(d,a);d\nend\nfunction factory(X,P1,P2,Y,D3)\n k0=stage0(CUDABackend(),64);k1=stage1(CUDABackend(),64);k2=stage2(CUDABackend(),64)\n function launch()\n  k0(X,P1,D3;ndrange=1048576)\n  k1(P1,P2,D3;ndrange=1048576)\n  k2(P2,Y,D3;ndrange=1048576)\n  nothing\n end\n launch\nend\ninclude("../study_common.jl")\nzmddft_study("ka",makearray,factory)\n'
 dest=P/'zmddft-julia-KA';dest.mkdir(exist_ok=True);f=dest/'main.jl';f.write_text(out)
 (P/'kernel_generation_ka.json').write_text(json.dumps(dict(source=str(source.relative_to(R)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reference_Julia_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),output=str(f.relative_to(R)),output_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),generated_Julia_bytes=len(out.encode()),authored_device_definitions=3,source_assignment_body_reused=True,shared_reference_generation_accounted_once=True),indent=2)+'\n')
 print('GeneratedKA',len(out.encode()));raise SystemExit
headers=list(re.finditer(r'__global__\s+void (ker_zmddft_fwd_256x256x256_cu\d)\(([^)]+)\)\s*\{',s));out='using CUDA\n# Generated from exact Spiral source by generate_kernels.py.\n'
counts=[]
for stage,m in enumerate(headers):
 start=m.end();end=s.index('\n}',start);body=s[start:end];params=re.findall(r'\*(\w+)',m.group(2));shared=re.search(r'__shared__ double (\w+)\[2048\];',body).group(1)
 body=re.sub(r'__shared__ double \w+\[2048\];','',body);body=re.sub(r'\b(?:double|int)\s+[^;]+;','',body)
 lines=[]
 for statement in body.split(';'):
  text=statement.strip()
  if not text:continue
  if text=='__syncthreads()':lines.append('CUDA.sync_threads()');continue
  text=text.replace('threadIdx.x','local0').replace('blockIdx.x','group0').replace(' / ',' ÷ ')
  text=re.sub(r'(?<![\w.])\d+(?![\w.])',lambda z:'Int32('+z.group(0)+')',text)
  text=re.sub(r'(\w+)\[([^\]]+)\]',lambda z:z.group(1)+'[('+z.group(2)+')+Int32(1)]',text)
  lines.append('@inbounds '+text)
 out+='function stage'+str(stage)+'('+','.join(params)+',D3)\n local0=Int32(threadIdx().x-1);group0=Int32(blockIdx().x-1)\n '+shared+'=CUDA.CuStaticSharedArray(Float64,2048)\n '+'\n '.join(lines)+'\n nothing\nend\n'
 counts.append(dict(stage=stage,source_kernel=m.group(1),statements=len(lines),barriers=lines.count('CUDA.sync_threads()'),shared_doubles=2048))
out+='''function factory(X,P1,P2,Y,D3)
 function launch()
  @cuda threads=64 blocks=16384 stage0(X,P1,D3)
  @cuda threads=64 blocks=16384 stage1(P1,P2,D3)
  @cuda threads=64 blocks=16384 stage2(P2,Y,D3)
  nothing
 end
 launch
end
include("../study_common.jl")
zmddft_study("cuda_jl",CuArray,factory)
'''
f=P/'zmddft-julia-cuda.jl/main.jl';f.write_text(out);(P/'kernel_generation_reference.json').write_text(json.dumps(dict(source=str(source.relative_to(R)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),output=str(f.relative_to(R)),output_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),method='Straightline source assignments/indexing/scope preserved; Czeroindices+1, Int32indexconstants, shared2048/barriers, noFFTlibraryshortcut',counts=counts,generated_Julia_bytes=len(out.encode()),generated_bytes_not_billed_tokens=True),indent=2)+'\n');print(counts)
