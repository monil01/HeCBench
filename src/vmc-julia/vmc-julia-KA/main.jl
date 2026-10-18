using CUDA,KernelAbstractions
include("../vmc_device.jl")
@kernel unsafe_indices=true function init!(seed,states);init_site!(@index(Global,Linear),seed,states);end
@kernel unsafe_indices=true function initialize!(a,states);initialize_site!(@index(Global,Linear),a,states);end
@kernel unsafe_indices=true function zero!(stats,N);zero_site!(@index(Global,Linear),stats,N);end
@kernel unsafe_indices=true function propagate!(steps,a,stats,states,N);propagate_site!(@index(Global,Linear),steps,a,stats,states,N);end
@kernel unsafe_indices=true function reduce!(n,data,offset,out,outputoffset,threads,grid)
 tid=@index(Local,Linear);bid=@index(Group,Linear);i=(bid-1)*threads+tid;stride=threads*grid;shared=@localmem Float32 (512,);total=0f0
 while i<=n;@inbounds total+=data[offset+i];i+=stride;end
 @inbounds shared[tid]=total
 @synchronize
 s=Int32(128)
 while s>0
  if tid<=s&&tid+s<=threads;@inbounds shared[tid]+=shared[tid+s];end
  @synchronize
  s>>=1
 end
 if tid==1;@inbounds out[outputoffset+bid]=shared[1];end
end
const ik=init!(CUDABackend(),256);const nk=initialize!(CUDABackend(),256);const zk=zero!(CUDABackend(),256);const pk=propagate!(CUDABackend(),256);const r1=reduce!(CUDABackend(),256);const r2=reduce!(CUDABackend(),224)
function reductions(stats,sums,blocks,N)
 for j in Int32(0):Int32(3)
  r1(N,stats,j*N,blocks,Int32(0),Int32(256),Int32(224);ndrange=57344)
  r2(Int32(224),blocks,Int32(0),sums,j,Int32(224),Int32(1);ndrange=224)
 end
end
launch=(init=(seed,states)->ik(seed,states;ndrange=57344),initialize=(a,states)->nk(a,states;ndrange=57344),zero=(stats,N)->zk(stats,N;ndrange=57344),propagate=(steps,a,stats,states,N)->pk(steps,a,stats,states,N;ndrange=57344),reduce=reductions)
include("../study_common.jl")
vmc_study(launch)
