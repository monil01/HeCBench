using CUDA,JACC
include("../study_portable_common.jl")
@inline function flags_kernel(i,input,flags,n)
 i=Int32(i)
 if i<=n;@inbounds flags[i]=(i==1||input[i]!=input[i-Int32(1)]) ? Int32(1) : Int32(0);end
 return
end
@inline function block_kernel(i,input,output,totals,n,template)
 tid=Int32((i-1)%256);bid=Int32((i-1)÷256);i=Int32(i);shared=JACC.shared(template)
 @inbounds shared[tid+Int32(1)]=i<=n ? input[i] : Int32(0)
 JACC.sync_workgroup();offset=Int32(1)
 while offset<Int32(256)
  @inbounds v=tid>=offset ? shared[tid-offset+Int32(1)] : Int32(0)
  JACC.sync_workgroup()
  @inbounds shared[tid+Int32(1)]+=v
  JACC.sync_workgroup();offset*=Int32(2)
 end
 if i<=n;@inbounds output[i]=shared[tid+Int32(1)];end
 if tid==255;@inbounds totals[bid+Int32(1)]=shared[256];end
 return
end
@inline function add_kernel(i,output,offsets,n)
 i=Int32(i);bid=(i-Int32(1))÷Int32(256)
 if i<=n&&bid>0;@inbounds output[i]+=offsets[bid];end
 return
end
@inline function scatter_kernel(i,input,flags,prefix,keys,starts,n)
 i=Int32(i)
 if i<=n
  if @inbounds flags[i]==1;@inbounds r=prefix[i];keys[r]=input[i];starts[r]=i;end
 end
 return
end
@inline function lengths_kernel(i,prefix,starts,lens,count,n)
 i=Int32(i);nr=@inbounds prefix[n]
 if i==1;@inbounds count[1]=nr;end
 if i<=nr;@inbounds finish=i==nr ? n+Int32(1) : starts[i+Int32(1)];lens[i]=finish-starts[i];end
 return
end
const BACKEND=JACC.get_backend(:cuda)
function factory(input,keys,lens,count,n)
 specs0=Dict{Int32,Any}();specs1=Dict{Int32,Any}();sz=n
 while true
  for (dic,bytes)in ((specs0,0),(specs1,1024))
   dic[Int32(sz)]=JACC.LaunchSpec{typeof(BACKEND)}(;threads=256,blocks=cld(sz,256),shmem_size=bytes,sync=false)
  end
  sz<=256&&break;sz=cld(sz,256)
 end
 template=CUDA.zeros(Int32,256)
 flags(a,b,n)=JACC.parallel_for(specs0[n],cld(n,256)*256,flags_kernel,a,b,n)
 block(a,b,t,n)=JACC.parallel_for(specs1[n],cld(n,256)*256,block_kernel,a,b,t,n,template)
 add(a,b,n)=JACC.parallel_for(specs0[n],cld(n,256)*256,add_kernel,a,b,n)
 scatter(a,b,c,d,e,n)=JACC.parallel_for(specs0[n],cld(n,256)*256,scatter_kernel,a,b,c,d,e,n)
 lengths(a,b,c,d,n)=JACC.parallel_for(specs0[n],cld(n,256)*256,lengths_kernel,a,b,c,d,n)
 make_encoder(input,keys,lens,count,n,flags,block,add,scatter,lengths)
end
rle_study(factory)
