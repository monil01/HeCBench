using CUDA,KernelAbstractions
include("../study_device.jl")
@kernel function base_kernel!(reads,total)
 i=@index(Global,Linear)
 base_body!(reads,total,Int32(i)-O)
end
@kernel function compress_kernel!(lengths,offsets,reads,compressed,gaps,n)
 i=@index(Global,Linear)
 compress_body!(lengths,offsets,reads,compressed,gaps,n,Int32(i)-O)
end
@kernel function index4_kernel!(reads,lengths,offsets,index,orders,words,magic,n)
 i=@index(Global,Linear)
 index_body!(reads,lengths,offsets,index,orders,words,magic,n,Int32(i)-O,Val(4))
end
@kernel function index5_kernel!(reads,lengths,offsets,index,orders,words,magic,n)
 i=@index(Global,Linear)
 index_body!(reads,lengths,offsets,index,orders,words,magic,n,Int32(i)-O,Val(5))
end
@kernel function index6_kernel!(reads,lengths,offsets,index,orders,words,magic,n)
 i=@index(Global,Linear)
 index_body!(reads,lengths,offsets,index,orders,words,magic,n,Int32(i)-O,Val(6))
end
@kernel function index7_kernel!(reads,lengths,offsets,index,orders,words,magic,n)
 i=@index(Global,Linear)
 index_body!(reads,lengths,offsets,index,orders,words,magic,n,Int32(i)-O,Val(7))
end
@kernel function cutoff_kernel!(threshold,k,lengths,words,cutoff,n)
 i=@index(Global,Linear)
 cutoff_body!(threshold,k,lengths,words,cutoff,n,Int32(i)-O)
end
@kernel function merge_kernel!(offsets,index,orders,words,n)
 i=@index(Global,Linear)
 merge_body!(offsets,index,orders,words,n,Int32(i)-O)
end
@kernel function rep_kernel!(cluster,rep,n)
 representative_body!(cluster,rep,n)
end
@kernel function make_kernel!(offsets,index,orders,words,table,rep)
 i=@index(Global,Linear)
 table_body!(offsets,index,orders,words,table,rep,Int32(i)-O,Val(false))
end
@kernel function clean_kernel!(offsets,index,orders,words,table,rep)
 i=@index(Global,Linear)
 table_body!(offsets,index,orders,words,table,rep,Int32(i)-O,Val(true))
end
@kernel function magic_kernel!(threshold,lengths,magic,cluster,rep,n)
 i=@index(Global,Linear)
 magic_body!(threshold,lengths,magic,cluster,rep,n,Int32(i)-O)
end
@kernel function align_kernel!(threshold,lengths,offsets,compressed,gaps,rep,cluster,n)
 i=@index(Global,Linear)
 align_body!(threshold,lengths,offsets,compressed,gaps,rep,cluster,n,Int32(i)-O)
end
@kernel function filter_kernel!(threshold,k,lengths,offsets,index,orders,words,cutoff,cluster,table,n)
 result=@localmem Int32 (128,)
 group=@index(Group,Linear);lane=@index(Local,Linear)
 active=filter_sum!(offsets,index,orders,words,cluster,table,n,result,Int32(group)-O,Int32(lane)-O)
 if active
  @synchronize
  filter_finish!(cutoff,cluster,result,Int32(group)-O,Int32(lane)-O)
 end
end
launch_base(c)=base_kernel!(CUDABackend(),128)(c.reads.data,Int64(c.total);ndrange=16384)
launch_compress(c)=compress_kernel!(CUDABackend(),128)(c.lengths.data,c.offsets.data,c.reads.data,c.compressed.data,c.gaps.data,c.n;ndrange=cld(Int(c.n),128)*128)
function launch_index(c)
 args=(c.reads.data,c.lengths.data,c.offsets.data,c.index.data,c.orders.data,c.words.data,c.magic.data,c.n)
 if c.k==4;index4_kernel!(CUDABackend(),128)(args...;ndrange=cld(Int(c.n),128)*128)
 elseif c.k==5;index5_kernel!(CUDABackend(),128)(args...;ndrange=cld(Int(c.n),128)*128)
 elseif c.k==6;index6_kernel!(CUDABackend(),128)(args...;ndrange=cld(Int(c.n),128)*128)
 else;index7_kernel!(CUDABackend(),128)(args...;ndrange=cld(Int(c.n),128)*128);end
 nothing
end
launch_cutoff(c)=cutoff_kernel!(CUDABackend(),128)(c.threshold,c.k,c.lengths.data,c.words.data,c.cutoff.data,c.n;ndrange=cld(Int(c.n),128)*128)
launch_merge(c)=merge_kernel!(CUDABackend(),128)(c.offsets.data,c.index.data,c.orders.data,c.words.data,c.n;ndrange=cld(Int(c.n),128)*128)
launch_rep(c)=rep_kernel!(CUDABackend(),1)(c.cluster.data,c.rep.data,c.n;ndrange=1)
launch_make(c,r)=make_kernel!(CUDABackend(),128)(c.offsets.data,c.index.data,c.orders.data,c.words.data,c.table.data,r;ndrange=16384)
launch_clean(c,r)=clean_kernel!(CUDABackend(),128)(c.offsets.data,c.index.data,c.orders.data,c.words.data,c.table.data,r;ndrange=16384)
launch_magic(c,r)=magic_kernel!(CUDABackend(),128)(c.threshold,c.lengths.data,c.magic.data,c.cluster.data,r,c.n;ndrange=cld(Int(c.n),128)*128)
launch_filter(c)=filter_kernel!(CUDABackend(),128)(c.threshold,c.k,c.lengths.data,c.offsets.data,c.index.data,c.orders.data,c.words.data,c.cutoff.data,c.cluster.data,c.table.data,c.n;ndrange=Int(c.n)*128)
launch_align(c,r)=align_kernel!(CUDABackend(),128)(c.threshold,c.lengths.data,c.offsets.data,c.compressed.data,c.gaps.data,r,c.cluster.data,c.n;ndrange=cld(Int(c.n),128)*128)
include("../study_common.jl")
main()
