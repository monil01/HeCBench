using CUDA,KernelAbstractions,Atomix
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function fpc_branch_ka!(values,counter)
 tid=Int32(@index(Local,Linear));i=Int32(@index(Global,Linear))
 scratch=@localmem UInt32 (1,)
 value=@inbounds values[i];inc=UInt32(fpc_increment(value))
 if tid==Int32(1);@inbounds scratch[1]=UInt32(0);end
 @synchronize
 Atomix.@atomic scratch[1]+=inc
 @synchronize
 if tid==Int32(256);Atomix.@atomic counter[1]+=scratch[1];end
end
@kernel function fpc_masks_ka!(values,counter)
 tid=Int32(@index(Local,Linear));i=Int32(@index(Global,Linear))
 scratch=@localmem UInt32 (1,)
 value=@inbounds values[i];inc=UInt32(fpc_increment_masks(value))
 if tid==Int32(1);@inbounds scratch[1]=UInt32(0);end
 @synchronize
 Atomix.@atomic scratch[1]+=inc
 @synchronize
 if tid==Int32(256);Atomix.@atomic counter[1]+=scratch[1];end
end
function factory(values,counter,n,masked)
 kernel=masked ? fpc_masks_ka! : fpc_branch_ka!
 launch=kernel(get_backend(values),256)
 return ()->launch(values,counter;ndrange=n)
end
fpc_study(factory)
