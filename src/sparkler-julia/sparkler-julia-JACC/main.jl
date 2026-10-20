using CUDA,JACC
const MODEL="jacc"
function set_input!(i,d,nr::UInt64,nc::UInt64,ld::UInt64,base::UInt64)
 index=UInt64(i-1)
 if index<nr*nc
  r=index%nr;c=index÷nr;s=UInt64(1)+(r+base)%UInt64(499)
  @inbounds d[Int(r+ld*c+1)]=c%s==0 ? 1f0 : 0f0
 end
 return nothing
end
function launch!(a,base)
 # Fresh mutable spec per shape; explicit zero dynamic shared storage and exact source grid.
 spec=JACC.launch_spec(stream=CUDA.stream(),threads=256,blocks=cld(a.rows*a.cols,256),shmem_size=0,sync=false)
 JACC.parallel_for(spec,a.rows*a.cols,set_input!,a.d,UInt64(a.rows),UInt64(a.cols),UInt64(a.ld),base)
end
include(joinpath(@__DIR__,"../study_common.jl"))
study()
