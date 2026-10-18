using CUDA
include("../vmc_device.jl")
@inline site()=(blockIdx().x-Int32(1))*blockDim().x+threadIdx().x
function init!(seed,states);init_site!(site(),seed,states);return;end
function initialize!(a,states);initialize_site!(site(),a,states);return;end
function zero!(stats,N);zero_site!(site(),stats,N);return;end
function propagate!(steps,a,stats,states,N);propagate_site!(site(),steps,a,stats,states,N);return;end
function reduce!(n,data,offset,out,outputoffset)
 tid=Int32(threadIdx().x);i=site();stride=Int32(blockDim().x*gridDim().x);shared=@cuStaticSharedMem(Float32,512);total=0f0
 while i<=n;@inbounds total+=data[offset+i];i+=stride;end
 @inbounds shared[tid]=total
 sync_threads();s=Int32(128)
 while s>0
  if tid<=s&&tid+s<=blockDim().x;@inbounds shared[tid]+=shared[tid+s];end
  sync_threads();s>>=1
 end
 if tid==1;@inbounds out[outputoffset+Int32(blockIdx().x)]=shared[1];end
 return
end
function reductions(stats,sums,blocks,N)
 for j in Int32(0):Int32(3)
  @cuda threads=256 blocks=224 reduce!(N,stats,j*N,blocks,Int32(0))
  @cuda threads=224 blocks=1 reduce!(Int32(224),blocks,Int32(0),sums,j)
 end
end
launch=(init=(seed,states)->(@cuda threads=256 blocks=224 init!(seed,states)),initialize=(a,states)->(@cuda threads=256 blocks=224 initialize!(a,states)),zero=(stats,N)->(@cuda threads=256 blocks=224 zero!(stats,N)),propagate=(steps,a,stats,states,N)->(@cuda threads=256 blocks=224 propagate!(steps,a,stats,states,N)),reduce=reductions)
include("../study_common.jl")
vmc_study(launch)
