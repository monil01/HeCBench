from pathlib import Path
P=Path(__file__).resolve().parents[1]
# All shared indices are zero-based in this source-reviewed mathematical template.
v='''for c in 0:5;s[c*96+tid+1]=Q[gid*576+c*96+tid+1];end
if tid<12;s[576+tid+1]=G[gid*12+tid+1];end
SYNC
'''
for c in range(6):
 for r in range(3):v+=f'd{c}{r}=0f0\n'
v+='for m in 0:83\n dr=D[4*(tid+m*96)+1];ds=D[4*(tid+m*96)+2];dt=D[4*(tid+m*96)+3]\n'
for c in range(6):
 v+=f'q=s[{c}*96+m+1]\n'
 for r,n in enumerate(['dr','ds','dt']):v+=f'd{c}{r}=muladd({n},q,d{c}{r})\n'
v+='end\n'
for ref,base in [('r',0),('s',4),('t',8)]:
 for a,axis in enumerate('xyz'):v+=f'{ref}{axis}=s[576+{base+a}+1]\n'
for field,sign,field1,axis1,field2,axis2 in [(0,-1,5,'y',4,'z'),(1,-1,3,'z',5,'x'),(2,-1,4,'x',3,'y'),(3,1,2,'y',1,'z'),(4,1,0,'z',2,'x'),(5,1,1,'x',0,'y')]:
 term='+'.join(f'{ref}{axis1}*d{field1}{r}'for r,ref in enumerate('rst'))+'-'+ '-'.join(f'{ref}{axis2}*d{field2}{r}'for r,ref in enumerate('rst'))
 v+=f'rhs[gid*576+{field}*96+tid+1]=Float32({sign}*({term}))\n'
(P/'translation_generators/volume_body.jl.txt').write_text(v)
s='''if tid<112
 off=gid*784+tid;m=Int(S[off+1]);p=Int(S[off+112+1]);Fsc=S[off+224+1];Bsc=S[off+336+1];nx=S[off+448+1];ny=S[off+560+1];nz=S[off+672+1]
 if p<0
  remote=6*(-1-p)
'''
for c in range(6):s+=f'  d{c}=Float64(Fsc*(part[remote+{c}+1]-Q[m+{c}*96+1]))\n'
s+=' else\n'
for c in range(6):s+=f'  d{c}=Float64(Fsc*('+('Bsc*'if c>=3 else'')+f'Q[p+{c}*96+1]-Q[m+{c}*96+1]))\n'
s+=''' end
 ndH=Float64(nx)*d0+Float64(ny)*d1+Float64(nz)*d2
 ndE=Float64(nx)*d3+Float64(ny)*d4+Float64(nz)*d5
'''
for c,expr in enumerate(['-ny*d5+nz*d4+d0-ndH*nx','-nz*d3+nx*d5+d1-ndH*ny','-nx*d4+ny*d3+d2-ndH*nz','ny*d2-nz*d1+d3-ndE*nx','nz*d0-nx*d2+d4-ndE*ny','nx*d1-ny*d0+d5-ndE*nz']):s+=f' s[{c}*112+tid+1]=Float32({expr})\n'
s+='end\nSYNC\nif tid<84\n'
for c in range(6):s+=f'r{c}=0f0\n'
s+='for m in 0:111\n l=L[4*(tid+84*div(m,4))+m%4+1]\n'
for c in range(6):s+=f'r{c}=muladd(l,s[{c}*112+m+1],r{c})\n'
s+='end\n'
for c in range(6):s+=f'rhs[gid*576+{c}*96+tid+1]+=r{c}\n'
s+='end\n';(P/'translation_generators/surface_body.jl.txt').write_text(s)
r='''if i<294336
 value=muladd(fa,res[i+1],fdt*rhs[i+1]);res[i+1]=value;Q[i+1]=muladd(fb,value,Q[i+1])
end
''';(P/'translation_generators/rk_body.jl.txt').write_text(r)
g='''if i<12768;out[i+1]=Q[Int(map[i+1])+1];end
''';(P/'translation_generators/gather_body.jl.txt').write_text(g)
code='using CUDA\n'
for name,args,shared,body in [('volume','Q,D,G,rhs',588,v),('surface','Q,part,L,S,rhs',672,s),('rk','Q,res,rhs,fa,fb,fdt',0,r),('gather','Q,map,out',0,g)]:
 code+=f'function {name}_kernel({args})\n tid=Int(threadIdx().x)-1;gid=Int(blockIdx().x)-1;i=gid*Int(blockDim().x)+tid\n'
 if shared:code+=f's=CUDA.@cuStaticSharedMem(Float32,{shared})\n'
 code+='@inbounds begin\n'+body.replace('SYNC','CUDA.sync_threads()')+'\nend\nreturn nothing\nend\n'
code+='''alloc_backend(::Type{T},n) where T=CUDA.CuArray{T}(undef,n)
run_gather!(Q,mapping,out)=CUDA.@cuda threads=256 blocks=50 gather_kernel(Q,mapping,out)
run_volume!(Q,D,G,rhs)=CUDA.@cuda threads=84 blocks=511 volume_kernel(Q,D,G,rhs)
run_surface!(Q,part,L,S,rhs)=CUDA.@cuda threads=112 blocks=511 surface_kernel(Q,part,L,S,rhs)
run_rk!(Q,res,rhs,fa,fb,fdt)=CUDA.@cuda threads=256 blocks=1150 rk_kernel(Q,res,rhs,fa,fb,fdt)
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
'''
(P/'miniDGS-julia-cuda.jl/main.jl').write_text(code)
