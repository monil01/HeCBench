using CUDA,JACC
JACC.@init_backend
function permute_jacc!(index,q,k,v,inp,total::Int32,tdim::Int32,nh::Int32,d::Int32)
 idx0=Int32(index)-Int32(1)
 if idx0<total
  c=nh*d;b=idx0÷(c*tdim);rest=idx0%(c*tdim);nh_=rest÷(tdim*d);rest=rest%(tdim*d);n=rest÷d;d_=rest%d;inp_idx=b*tdim*Int32(3)*c+n*Int32(3)*c+nh_*d+d_+Int32(1);out_idx=idx0+Int32(1)
  @inbounds q[out_idx]=inp[inp_idx]
  @inbounds k[out_idx]=inp[inp_idx+c]
  @inbounds v[out_idx]=inp[inp_idx+Int32(2)*c]
 end
 return
end
function factory(out,inp,B,T,C,NH,bs)
 total=B*T*C;q=view(out,1:total);k=view(out,total+1:2total);v=view(out,2total+1:3total);spec=JACC.launch_spec(;threads=bs,shmem_size=0,sync=false)
 ()->begin;JACC.parallel_for(spec,total,permute_jacc!,q,k,v,inp,Int32(total),Int32(T),Int32(NH),Int32(C÷NH));CUDA.device_synchronize(blocking=true);end
end
include(joinpath(@__DIR__,"..","study_common.jl"))
permutation_study(factory)
