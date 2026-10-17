using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
# Scratch/specresources outsidealltimers; source-levelshared() initialcopy isnotlogicalinput.
const TEMPLATE_I=CUDA.zeros(Int32,256)
const TEMPLATE_F=(CUDA.zeros(Float32,8),CUDA.zeros(Float32,16),CUDA.zeros(Float32,32))
const BACKEND=JACC.get_backend(:cuda)
const SPEC_B=JACC.LaunchSpec{typeof(BACKEND)}(;threads=256,blocks=1,shmem_size=1024,sync=false)
const SPEC_R=ntuple(i->JACC.LaunchSpec{typeof(BACKEND)}(;threads=STUDY_SGS[i],blocks=STUDY_N÷STUDY_SGS[i],shmem_size=4STUDY_SGS[i],sync=false),3)
@inline function xor_jacc!(i,mode::Val{S},out,template) where S
 lid=i;lane=lid-1;cache=JACC.shared(template)
 value=Int32(lane&(S-1));@inbounds cache[lid]=value
 JACC.sync_workgroup()
 for mask in (1,2,4,8,16)
  if mask<S-1
   peer=(lane÷32)*32+((lane%32)⊻mask)+1
   @inbounds other=cache[peer]
   JACC.sync_workgroup()
   value+=other;@inbounds cache[lid]=value
   JACC.sync_workgroup()
  end
 end
 @inbounds out[i]=value
 return
end
@inline function broadcast_jacc!(i,mode::Val{S},arg,out,template) where S
 lid=i;lane=lid-1;cache=JACC.shared(template)
 @inbounds cache[lid]=(lane&(S-1))==0 ? reinterpret(Int32,arg) : Int32(0)
 JACC.sync_workgroup()
 @inbounds out[i]=cache[(lane÷32)*32+1]
 return
end
@inline function reverse_jacc!(i,mode::Val{S},out,input,template) where S
 lid=(i-1)%S+1;cache=JACC.shared(template)
 @inbounds cache[lid]=input[i]
 JACC.sync_workgroup()
 @inbounds out[i]=cache[S-lid+1]
 return
end
function launch!(kind,s,pattern,out,input)
 if kind==1
  JACC.parallel_for(xor_jacc!,SPEC_B,256,Val(s),out,TEMPLATE_I)
 elseif kind==2
  JACC.parallel_for(broadcast_jacc!,SPEC_B,256,Val(s),pattern,out,TEMPLATE_I)
 else
  which=s==8 ? 1 : s==16 ? 2 : 3
  JACC.parallel_for(reverse_jacc!,SPEC_R[which],STUDY_N,Val(s),out,input,TEMPLATE_F[which])
 end
end
study_main(launch!)
