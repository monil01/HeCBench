#!/usr/bin/env python3
"""Own KA entry derived from accepted CUDAjl topology; lexical KA barriers.
Does not modify any accepted CUDAjl/shared or native sources.
"""
from pathlib import Path
B=Path(__file__).resolve().parent;s=(B/'nbnxm-julia-cuda.jl/main.jl').read_text();start=s.index('function nb_entry!');end=s.index('function nb_launch_phase',start);body=s[start:end]
body=body.replace('function nb_entry!','@kernel function nb_entry!',1)
body=body.replace('    ix=Int32(threadIdx().x)-Int32(1);jy=Int32(threadIdx().y)-Int32(1)\n    block=Int32(blockIdx().z)-Int32(1)','    local_index=@index(Local,Cartesian);group_index=@index(Group,Cartesian)\n    ix=Int32(local_index[1])-Int32(1);jy=Int32(local_index[2])-Int32(1)\n    block=Int32(group_index[3])-Int32(1)')
for name,typ,count in [('smx','Float32',256),('smt','Int32',64),('red','Float32',192)]:body=body.replace(f'{name}=CuStaticSharedArray({typ},{count})',f'{name}=@localmem {typ} ({count},)')
head='    Base.Cartesian.@nexprs 8 c -> begin\n';st=body.index(head);fin=body.index('    if doShift && jy<Int32(3)',st);segment=body[st+len(head):fin];assert segment.endswith('    end\n');segment=segment[:-len('    end\n')];expanded=''
for c in range(1,9):expanded+=f'    # Source-unrolled cluster {c-1}: lexical barriers visible to KA.\n'+segment.replace('[c]',f'[{c}]').replace('Int32(c-1)',f'Int32({c-1})')
body=body[:st]+expanded+body[fin:];body=body.replace('sync_threads()','@synchronize')
text='using CUDA,Libdl,SHA,KernelAbstractions\ninclude(joinpath(@__DIR__,"..","nb_device.jl"))\nconst NB_KA_BACKEND=CUDA.CUDABackend(always_inline=true)\n'+body+'''function nb_launch_phase(q,a)
    kernel=nb_entry!(NB_KA_BACKEND,(8,8,1))
    for repetition in 1:Int(q.repeat)
        kernel(a...,q.numTypes,q.cutoffSq,q.beta,q.epsFac,q.calcShift!=0;ndrange=(8,8,Int(q.gridZ)))
    end
    nothing
end
include(joinpath(@__DIR__,"..","study_common.jl"))
nb_study()
''';D=B/'nbnxm-julia-KA';D.mkdir(exist_ok=True);(D/'main.jl').write_text(text)
