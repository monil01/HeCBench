from pathlib import Path
P=Path('/home/imo/HeCBench_julia/src/mmcsf-julia');out=['using CUDA,Libdl,SHA','include(joinpath(@__DIR__,"device_bodies.jl"))']
# Explicit seven wrappers preserve each original kernel parameter order.
for op in range(7):
 four=op in [1,2,4,6];args='v,l,i,p0,p1,i1,'+('p2,i2,'if four else'')+'nf,u0,u1,u2,'+('u3,'if four else'')+'mode,r,w,log'+(',fpw,lfpw'if op in[0,1]else'')
 out += [f'function mm_kernel{op}({args})',f'    mm_body{op}({args},Int64(threadIdx().x)-1,Int64(blockIdx().x)-1,Int64(blockDim().x))','    return','end']
out+=['function mm_launch(q,a)','    op=Int(q.op);0<=op<7||error("UnknownMMop")']
for op in range(7):
 four=op in[1,2,4,6];arg='a[1],a[2],a[3],a[4],a[5],a[6],'+('a[7],a[8],'if four else'')+'Int64(q.nfibers),a[9],a[10],a[11],'+('a[12],'if four else'')+'Int64(q.mode),Int64(q.rank),Int64(q.wps),Int64(q.logWPC)'+(',Int64(q.fpw),Int64(q.logFPW)'if op in[0,1]else'')
 out += [('    if'if op==0 else'    elseif')+f' op=={op}',f'        @cuda threads=Int(q.block) blocks=Int(q.grid) stream=MM_STREAM mm_kernel{op}({arg})']
out+=['    end','    nothing','end','include(joinpath(@__DIR__,"..","secondary_fp64_atomic","study_common.jl"))','mm_study()'];(P/'mmcsf-julia-cuda.jl/main.jl').write_text('\n'.join(out)+'\n')
