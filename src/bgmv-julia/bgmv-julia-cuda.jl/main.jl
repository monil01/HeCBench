using CUDA
include(joinpath(@__DIR__,"..","study_common.jl"))
function kernel!(output,input,weights,indices,::Val{BS},::Val{MODE}) where {BS,MODE}
 tid=threadIdx().x-Int32(1);token=blockIdx().x-Int32(1);channel=blockIdx().y-Int32(1);tmp=CuStaticSharedArray(Float32,BS);tmp[tid+1]=partial(tid,token,channel,input,weights,indices,Val(BS),Val(MODE));sync_threads();stride=Int32(BS÷2)
 while stride>0
  if tid<stride;@inbounds tmp[tid+1]+=tmp[tid+stride+1];end
  sync_threads();stride÷=Int32(2)
 end
 if tid==0;@inbounds store_result(output,token,channel,tmp[1],Val(MODE));end
 return
end
run!(out,input,w,indices,tmp,::Val{BS},::Val{MODE}) where {BS,MODE}=(@cuda threads=BS blocks=(TT,MODE<=2 ? RR : HH) kernel!(out,input,w,indices,Val(BS),Val(MODE));nothing)
study(run!)
