using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(indices,counts,scores,threshold,priors,topk)
 tid=threadIdx().x-Int32(1);group=(blockIdx().y-Int32(1))*Int32(CLASSES)+blockIdx().x-Int32(1);bins=CuStaticSharedArray(Int32,2048)
 for part in Int32(0):Int32(7);@inbounds bins[part*Int32(256)+tid+1]=Int32(0);end
 sync_threads();histogram_body(tid,group,scores,bins,threshold,priors);sync_threads();suffix_body(tid,bins)
 if tid==0;@inbounds counts[group+1]=Int32(0);end
 sync_threads();select_body(tid,group,scores,indices,counts,bins,threshold,priors,topk)
 return
end
run!(indices,counts,scores,tmp,threshold,priors,topk)=(@cuda threads=256 blocks=(CLASSES,BATCH) kernel!(indices,counts,scores,threshold,priors,topk);nothing)
study(run!)
