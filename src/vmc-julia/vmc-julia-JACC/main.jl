using CUDA,JACC
include("../vmc_device.jl")
function init!(i,seed,states);init_site!(i,seed,states);nothing;end
function initialize!(i,a,states);initialize_site!(i,a,states);nothing;end
function zero!(i,stats,N);zero_site!(i,stats,N);nothing;end
function propagate!(i,steps,a,stats,states,N);propagate_site!(i,steps,a,stats,states,N);nothing;end
function reduce!(idx,n,data,offset,out,outputoffset,threads,grid,template)
 tid=rem(Int32(idx)-Int32(1),threads)+Int32(1);bid=div(Int32(idx)-Int32(1),threads)+Int32(1);i=Int32(idx);stride=threads*grid;shared=JACC.shared(template);total=0f0
 while i<=n;@inbounds total+=data[offset+i];i+=stride;end
 @inbounds shared[tid]=total
 JACC.sync_workgroup();s=Int32(128)
 while s>0
  if tid<=s&&tid+s<=threads;@inbounds shared[tid]+=shared[tid+s];end
  JACC.sync_workgroup();s>>=1
 end
 if tid==1;@inbounds out[outputoffset+bid]=shared[1];end
 nothing
end
const point_spec=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=224,shmem_size=0,sync=false)
const reduce_spec1=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=256,blocks=224,shmem_size=2048,sync=false)
const reduce_spec2=JACC.LaunchSpec{typeof(JACC.get_backend(:cuda))}(;threads=224,blocks=1,shmem_size=2048,sync=false)
const template=CUDA.zeros(Float32,512)
function reductions(stats,sums,blocks,N)
 for j in Int32(0):Int32(3)
  JACC.parallel_for(reduce!,reduce_spec1,57344,N,stats,j*N,blocks,Int32(0),Int32(256),Int32(224),template)
  JACC.parallel_for(reduce!,reduce_spec2,224,Int32(224),blocks,Int32(0),sums,j,Int32(224),Int32(1),template)
 end
end
launch=(init=(seed,states)->JACC.parallel_for(init!,point_spec,57344,seed,states),initialize=(a,states)->JACC.parallel_for(initialize!,point_spec,57344,a,states),zero=(stats,N)->JACC.parallel_for(zero!,point_spec,57344,stats,N),propagate=(steps,a,stats,states,N)->JACC.parallel_for(propagate!,point_spec,57344,steps,a,stats,states,N),reduce=reductions)
include("../study_common.jl")
vmc_study(launch)
