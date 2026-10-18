from pathlib import Path
import re
p=Path('/home/imo/HeCBench_julia/src/miniWeather-julia');d=p/'miniWeather-julia-cuda.jl'
s='using CUDA\ninclude(joinpath(@__DIR__,"..","study_device.jl"))\n'
names=[('zhalo','update_state_z_body!',2),('fx','compute_flux_x_body!',2),('tx','compute_tend_x_body!',3),('fz','compute_flux_z_body!',2),('tz','compute_tend_z_body!',3),('update','update_fluid_state_body!',3),('pack','pack_body!',3),('unpack','unpack_body!',3)]
body=(p/'study_device.jl').read_text()
for alias,name,nd in names:
 match=re.search(r'function '+re.escape(name)+r'\(([^\n]+)\)',body);args=[x.strip()for x in match.group(1).split(',')];coordcount=2 if nd==2 else3;actual=args[:-coordcount];coords=args[-coordcount:];s+='function '+alias+'!('+','.join(actual)+')\n'
 for coord,dim in zip(coords,['x','y','z']):s+=' '+coord.split('::')[0]+'=(Int32(CUDA.blockIdx().'+dim+')-1)*Int32(CUDA.blockDim().'+dim+')+Int32(CUDA.threadIdx().'+dim+')-1\n'
 s+=' '+name+'('+','.join(x.split('::')[0]for x in args)+')\n return\nend\n'
s+='''function mass!(mass,te,state,h0,h1,nx::Int32,nz::Int32,dx::Float64,dz::Float64)
 i=(Int32(CUDA.blockIdx().x)-1)*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-1
 k=(Int32(CUDA.blockIdx().y)-1)*Int32(CUDA.blockDim().y)+Int32(CUDA.threadIdx().y)-1
 if i<nx && k<nz
  m,e=mass_terms_body(state,h0,h1,nx,nz,dx,dz,i,k)
  CUDA.@atomic mass[1]+=m
  CUDA.@atomic te[1]+=e
 end
 return
end
function launch(which,args,blocks,threads)
'''
for i,(alias,_,_)in enumerate(names+[('mass','',2)]):s+=(' if 'if i==0 else' elseif ')+'which==:'+alias+'\n  CUDA.@cuda threads=threads blocks=blocks '+alias+'!(args...)\n'
s+=' else;error("Unknown sourcekernel");end\nend\ninclude(joinpath(@__DIR__,"..","study_common.jl"))\nstudy((T,n)->CUDA.zeros(T,n),launch)\n';(d/'main.jl').write_text(s)
