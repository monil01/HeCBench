using CUDA,KernelAbstractions
import Atomix
@kernel unsafe_indices=true function portable_scatter!(indexes,source,outputs,n::Int32,o::Int32,v::Int32,::Val{MODE},::Val{NOUT}) where {MODE,NOUT}
 tid=Int32(@index(Local,Linear))-Int32(1);lane=tid%Int32(32);warp=tid÷Int32(32);base=(Int32(@index(Group,Linear))-Int32(1))*Int32(8192);nout=o*v
 outbuf=@localmem(Float32,(NOUT,));idxbuf=@localmem(Int32,(512,))
 pos=tid
 while pos<nout;@inbounds outbuf[pos+Int32(1)]=0f0;pos+=Int32(512);end
 if MODE==1;@synchronize();end
 start=Int32(0)
 while start<Int32(8192)
  bi=base+start+tid
  if bi<n;@inbounds idxbuf[tid+Int32(1)]=indexes[bi+Int32(1)];end
  @synchronize();bs=Int32(0)
  while bs<Int32(512)
   slot=bs+warp;bi=base+start+slot
   if bi<n
    index=@inbounds idxbuf[slot+Int32(1)];j=lane
    while j<v
     value=@inbounds source[bi*v+j+Int32(1)];Atomix.@atomic :monotonic outbuf[index*v+j+Int32(1)]+=value;j+=Int32(32)
    end
   end
   bs+=Int32(16)
  end
  @synchronize();start+=Int32(512)
 end
 pos=tid
 while pos<nout
  value=@inbounds outbuf[pos+Int32(1)];Atomix.@atomic :monotonic outputs[pos+Int32(1)]+=value;pos+=Int32(512)
 end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const PORTABLE=portable_scatter!(CUDABackend(),512)
function launch!(f)
 n,o,v=f[6:8];blocks=Int(n)÷8192+1
 PORTABLE(f[1],f[2],f[4],n,o,v,Val(f[9]),Val(Int(o)*Int(v));ndrange=blocks*512)
end
study_main("ka",launch!)
