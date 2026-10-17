using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function kernel!(indices,counts,scores,threshold,priors,topk)
 lane=@index(Local,Linear);groupidx=@index(Group,NTuple);tid=Int32(lane)-Int32(1);group=(Int32(groupidx[2])-Int32(1))*Int32(CLASSES)+Int32(groupidx[1])-Int32(1)
 bins=@localmem Int32 (2048,)
 for part in Int32(0):Int32(7);@inbounds bins[part*Int32(256)+tid+1]=Int32(0);end
 @synchronize
 histogram_body(tid,group,scores,bins,threshold,priors)
 @synchronize
 suffix_body(tid,bins)
 if lane==1;@inbounds counts[group+1]=Int32(0);end
 @synchronize
 select_body(tid,group,scores,indices,counts,bins,threshold,priors,topk)
end
run!(indices,counts,scores,tmp,threshold,priors,topk)=(kernel!(get_backend(indices),(256,1))(indices,counts,scores,threshold,priors,topk;ndrange=(256CLASSES,BATCH));nothing)
study(run!)
