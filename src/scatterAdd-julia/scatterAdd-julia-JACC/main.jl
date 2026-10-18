using CUDA,JACC
import Atomix
function portable_scatter!(i,indexes,source,outputs,n::Int32,o::Int32,v::Int32,::Val{MODE},template) where MODE
 gid=Int32(i)-Int32(1);tid=gid%Int32(512);lane=tid%Int32(32);warp=tid÷Int32(32);base=(gid÷Int32(512))*Int32(8192);nout=o*v
 outbuf=JACC.shared(template)
 pos=tid
 while pos<nout;@inbounds outbuf[pos+Int32(1)]=0f0;pos+=Int32(512);end
 if MODE==1;JACC.sync_workgroup();end
 start=Int32(0)
 while start<Int32(8192)
  bi=base+start+tid
  if bi<n;@inbounds outbuf[nout+tid+Int32(1)]=Float32(indexes[bi+Int32(1)]);end
  JACC.sync_workgroup();bs=Int32(0)
  while bs<Int32(512)
   slot=bs+warp;bi=base+start+slot
   if bi<n
    index=Int32(@inbounds outbuf[nout+slot+Int32(1)]);j=lane
    while j<v
     value=@inbounds source[bi*v+j+Int32(1)];Atomix.@atomic :monotonic outbuf[index*v+j+Int32(1)]+=value;j+=Int32(32)
    end
   end
   bs+=Int32(16)
  end
  JACC.sync_workgroup();start+=Int32(512)
 end
 pos=tid
 while pos<nout
  value=@inbounds outbuf[pos+Int32(1)];Atomix.@atomic :monotonic outputs[pos+Int32(1)]+=value;pos+=Int32(512)
 end
 return
end
include(joinpath(@__DIR__,"..","study_common.jl"))
const BACKEND=JACC.get_backend(:cuda)
const TEMPLATES=Dict{Int,Any}()
const SPECS=Dict{Tuple{Int,Int},Any}()
function launch!(f)
 n,o,v=f[6:8];blocks=Int(n)÷8192+1;nout=Int(o)*Int(v)
 template=get!(TEMPLATES,nout) do
  CUDA.zeros(Float32,nout+512)
 end
 spec=get!(SPECS,(nout,blocks)) do
  JACC.LaunchSpec{typeof(BACKEND)}(;threads=512,blocks=blocks,shmem_size=4*(nout+512),sync=false)
 end
 JACC.parallel_for(portable_scatter!,spec,blocks*512,f[1],f[2],f[4],n,o,v,Val(f[9]),template)
end
study_main("jacc",launch!)
