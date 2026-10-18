using CUDA
include("../study_common.jl")
function flags_kernel(input,flags,n)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)
 if i<=n;@inbounds flags[i]=(i==1||input[i]!=input[i-Int32(1)]) ? Int32(1) : Int32(0);end
 return
end
function block_kernel(input,output,totals,n)
 tid=Int32(threadIdx().x)-Int32(1);bid=Int32(blockIdx().x)-Int32(1);i=bid*Int32(256)+tid+Int32(1);shared=CuStaticSharedArray(Int32,256)
 @inbounds shared[tid+Int32(1)]=i<=n ? input[i] : Int32(0)
 sync_threads();offset=Int32(1)
 while offset<Int32(256)
  @inbounds v=tid>=offset ? shared[tid-offset+Int32(1)] : Int32(0)
  sync_threads()
  @inbounds shared[tid+Int32(1)]+=v
  sync_threads();offset*=Int32(2)
 end
 if i<=n;@inbounds output[i]=shared[tid+Int32(1)];end
 if tid==255;@inbounds totals[bid+Int32(1)]=shared[256];end
 return
end
function add_kernel(output,offsets,n)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x);bid=(i-Int32(1))÷Int32(256)
 if i<=n&&bid>0;@inbounds output[i]+=offsets[bid];end
 return
end
function scatter_kernel(input,flags,prefix,keys,starts,n)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x)
 if i<=n
  if @inbounds flags[i]==1;@inbounds r=prefix[i];keys[r]=input[i];starts[r]=i;end
 end
 return
end
function lengths_kernel(prefix,starts,lens,count,n)
 i=(Int32(blockIdx().x)-Int32(1))*Int32(256)+Int32(threadIdx().x);nr=@inbounds prefix[n]
 if i==1;@inbounds count[1]=nr;end
 if i<=nr;@inbounds finish=i==nr ? n+Int32(1) : starts[i+Int32(1)];lens[i]=finish-starts[i];end
 return
end
function factory(input,keys,lens,count,n)
 flags(a,b,n)=(@cuda threads=256 blocks=cld(n,256) flags_kernel(a,b,n))
 block(a,b,t,n)=(@cuda threads=256 blocks=cld(n,256) block_kernel(a,b,t,n))
 add(a,b,n)=(@cuda threads=256 blocks=cld(n,256) add_kernel(a,b,n))
 scatter(a,b,c,d,e,n)=(@cuda threads=256 blocks=cld(n,256) scatter_kernel(a,b,c,d,e,n))
 lengths(a,b,c,d,n)=(@cuda threads=256 blocks=cld(n,256) lengths_kernel(a,b,c,d,n))
 make_encoder(input,keys,lens,count,n,flags,block,add,scatter,lengths)
end
rle_study(factory)
