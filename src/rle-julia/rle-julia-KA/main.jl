using CUDA,KernelAbstractions
include("../study_portable_common.jl")
@kernel unsafe_indices=true function flags_kernel(input,flags,n)
 i=(Int32(@index(Group,Linear))-Int32(1))*Int32(256)+Int32(@index(Local,Linear))
 if i<=n;@inbounds flags[i]=(i==1||input[i]!=input[i-Int32(1)]) ? Int32(1) : Int32(0);end
end
@kernel unsafe_indices=true function block_kernel(input,output,totals,n)
 tid=Int32(@index(Local,Linear))-Int32(1);bid=Int32(@index(Group,Linear))-Int32(1);i=bid*Int32(256)+tid+Int32(1);shared=@localmem Int32 (256,)
 @inbounds shared[tid+Int32(1)]=i<=n ? input[i] : Int32(0)
 @synchronize;offset=Int32(1)
 while offset<Int32(256)
  @inbounds v=tid>=offset ? shared[tid-offset+Int32(1)] : Int32(0)
  @synchronize
  @inbounds shared[tid+Int32(1)]+=v
  @synchronize;offset*=Int32(2)
 end
 if i<=n;@inbounds output[i]=shared[tid+Int32(1)];end
 if tid==255;@inbounds totals[bid+Int32(1)]=shared[256];end
end
@kernel unsafe_indices=true function add_kernel(output,offsets,n)
 i=(Int32(@index(Group,Linear))-Int32(1))*Int32(256)+Int32(@index(Local,Linear));bid=(i-Int32(1))÷Int32(256)
 if i<=n&&bid>0;@inbounds output[i]+=offsets[bid];end
end
@kernel unsafe_indices=true function scatter_kernel(input,flags,prefix,keys,starts,n)
 i=(Int32(@index(Group,Linear))-Int32(1))*Int32(256)+Int32(@index(Local,Linear))
 if i<=n
  if @inbounds flags[i]==1;@inbounds r=prefix[i];keys[r]=input[i];starts[r]=i;end
 end
end
@kernel unsafe_indices=true function lengths_kernel(prefix,starts,lens,count,n)
 i=(Int32(@index(Group,Linear))-Int32(1))*Int32(256)+Int32(@index(Local,Linear));nr=@inbounds prefix[n]
 if i==1;@inbounds count[1]=nr;end
 if i<=nr;@inbounds finish=i==nr ? n+Int32(1) : starts[i+Int32(1)];lens[i]=finish-starts[i];end
end
function factory(input,keys,lens,count,n)
 backend=CUDABackend();kf=flags_kernel(backend,256);kb=block_kernel(backend,256);ka=add_kernel(backend,256);ks=scatter_kernel(backend,256);kl=lengths_kernel(backend,256)
 flags(a,b,n)=kf(a,b,n;ndrange=cld(n,256)*256)
 block(a,b,t,n)=kb(a,b,t,n;ndrange=cld(n,256)*256)
 add(a,b,n)=ka(a,b,n;ndrange=cld(n,256)*256)
 scatter(a,b,c,d,e,n)=ks(a,b,c,d,e,n;ndrange=cld(n,256)*256)
 lengths(a,b,c,d,n)=kl(a,b,c,d,n;ndrange=cld(n,256)*256)
 make_encoder(input,keys,lens,count,n,flags,block,add,scatter,lengths)
end
rle_study(factory)
