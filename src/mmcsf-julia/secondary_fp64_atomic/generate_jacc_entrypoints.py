from pathlib import Path
import re
P=Path('/home/imo/HeCBench_julia/src/mmcsf-julia');s=(P/'mmcsf-julia-cuda.jl/main.jl').read_text();s=s.replace('using CUDA,Libdl,SHA','using CUDA,Libdl,SHA,JACC\nconst MM_JACC_SPECS=Dict{Tuple{Int,Int,Int},Any}()');s=s.replace('include(joinpath(@__DIR__,"device_bodies.jl"))','include(joinpath(@__DIR__,"..","mmcsf-julia-cuda.jl","device_bodies.jl"))');s=re.sub(r'function mm_kernel(\d)\(',r'function mm_kernel\1(unused_index,',s);s=s.replace('    op=Int(q.op);0<=op<7||error("UnknownMMop")','''    op=Int(q.op);0<=op<7||error("UnknownMMop")
    spec=get!(MM_JACC_SPECS,(op,Int(q.grid),Int(q.block))) do
        JACC.launch_spec(;threads=Int(q.block),blocks=Int(q.grid),shmem_size=0,sync=false,stream=MM_STREAM)
    end''');s=re.sub(r'@cuda threads=Int\(q.block\) blocks=Int\(q.grid\) stream=MM_STREAM mm_kernel(\d)\(([^\n]*)\)',r'JACC.parallel_for(spec,Int(q.grid)*Int(q.block),mm_kernel\1,\2)',s);(P/'mmcsf-julia-JACC/main.jl').write_text(s)
