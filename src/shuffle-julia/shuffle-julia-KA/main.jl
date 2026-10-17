using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
# Fullgroupsbycontract; explicitindices preserve barrier-localregisterstate.
@kernel unsafe_indices=true function xor_ka!(mode::Val{S},out) where S
 lid=@index(Local,Linear);i=@index(Global,Linear);lane=lid-1
 cache=@localmem Int32 (256,)
 value=Int32(lane&(S-1));@inbounds cache[lid]=value
 @synchronize
 for mask in (1,2,4,8,16)
  if mask<S-1
   peer=(lane÷32)*32+((lane%32)⊻mask)+1
   @inbounds other=cache[peer]
   @synchronize
   value+=other;@inbounds cache[lid]=value
   @synchronize
  end
 end
 @inbounds out[i]=value
end
@kernel unsafe_indices=true function broadcast_ka!(mode::Val{S},arg,out) where S
 lid=@index(Local,Linear);i=@index(Global,Linear);lane=lid-1
 cache=@localmem Int32 (256,)
 @inbounds cache[lid]=(lane&(S-1))==0 ? reinterpret(Int32,arg) : Int32(0)
 @synchronize
 @inbounds out[i]=cache[(lane÷32)*32+1]
end
@kernel unsafe_indices=true function reverse_ka!(mode::Val{S},out,input) where S
 lid=@index(Local,Linear);i=@index(Global,Linear)
 cache=@localmem Float32 (S,)
 @inbounds cache[lid]=input[i]
 @synchronize
 @inbounds out[i]=cache[S-lid+1]
end
function launch!(kind,s,pattern,out,input)
 backend=KernelAbstractions.get_backend(out)
 if kind==1
  xor_ka!(backend,256)(Val(s),out;ndrange=256)
 elseif kind==2
  broadcast_ka!(backend,256)(Val(s),pattern,out;ndrange=256)
 else
  reverse_ka!(backend,s)(Val(s),out,input;ndrange=STUDY_N)
 end
end
study_main(launch!)
