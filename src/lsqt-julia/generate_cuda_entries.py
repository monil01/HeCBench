from pathlib import Path
p=Path('/home/imo/HeCBench_julia/src/lsqt-julia');d=p/'lsqt-julia-cuda.jl';d.mkdir(exist_ok=True)
entries=[('zero','n,re,im'),('copy','n,ar,ai,br,bi'),('add','n,ar,ai,br,bi'),('sz','n,ar,ai,br,bi'),('ip1','n,ar,ai,br,bi,orr,oi,offset'),('ip2','n,ar,ai,orr,oi'),('H','n,emax,nn,nl,pot,hr,hi,ar,ai,orr,oi'),('commutator','n,emax,nn,nl,hr,hi,dx,ar,ai,orr,oi'),('current','n,nn,nl,hr,hi,dx,ar,ai,orr,oi'),('cheb01','n,r0,i0,r1,i1,orr,oi,b0,b1,direction'),('cheb2','n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2,orr,oi,b,label'),('cheb1x','n,r1x,i1x,orr,oi,b'),('cheb2x','n,emax,nn,nl,pot,hr,hi,dx,r0,i0,r0x,i0x,r1,i1,r1x,i1x,r2,i2,r2x,i2x,orr,oi,b,label'),('poly','n,emax,nn,nl,pot,hr,hi,r0,i0,r1,i1,r2,i2')]
s='''# GPUQT Julia translation; originalGPLv3-or-later attribution in ../original_native/LICENCE.
using CUDA
const MODEL="cuda_jl"
const BACKEND=Val(:cuda_jl)
@inline block_sync(::Val{:cuda_jl})=CUDA.sync_threads()
@inline warp_sync(::Val{:cuda_jl})=CUDA.sync_warp(UInt32(0xffffffff))
include("../study_device.jl")
'''
for name,args in entries:
 s+=f'function kernel_{name}!({args})\n'
 if name in ['ip1','ip2']:
  s+=' tid=Int32(threadIdx().x)-Int32(1);block=Int32(blockIdx().x)-Int32(1)\n sr=CUDA.CuStaticSharedArray(Float64,256);si=CUDA.CuStaticSharedArray(Float64,256)\n'
  s+=f' body_{name}!(tid,block,{args},sr,si,BACKEND)\n'
 else:s+=f' i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)-Int32(1)\n body_{name}!(i,{args})\n'
 s+=' nothing\nend\n'
for code,(name,args)in enumerate(entries):
 s+=f'function dispatch!(::Val{{:{name}}},s,blocks,{args})\n record_launch!(s,{code})\n'
 if name=='cheb01':s+=' record_scalar!(s,9,(b0,b1,Float64(direction)))\n'
 if name=='cheb2':s+=' record_scalar!(s,10,(b,Float64(label)))\n'
 if name=='cheb1x':s+=' record_scalar!(s,11,(b,))\n'
 if name=='cheb2x':s+=' record_scalar!(s,12,(b,Float64(label)))\n'
 s+=f' CUDA.@cuda threads=256 blocks=blocks kernel_{name}!({args})\n nothing\nend\n'
s+='include("../study_common.jl")\nmain()\n';(d/'main.jl').write_text(s)
# Preserve generator: source generation invocation is not an independent mathematical candidate.
(p/'generate_cuda_entries.py').write_text(Path(__file__).read_text())
