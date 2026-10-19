using CUDA
import JACC
JACC.@init_backend
include("../study_device.jl")
function base_kernel!(i,args...);base_body!(args...,Int32(i)-O);return;end
function compress_kernel!(i,args...);compress_body!(args...,Int32(i)-O);return;end
function index4_kernel!(i,args...);index_body!(args...,Int32(i)-O,Val(4));return;end
function index5_kernel!(i,args...);index_body!(args...,Int32(i)-O,Val(5));return;end
function index6_kernel!(i,args...);index_body!(args...,Int32(i)-O,Val(6));return;end
function index7_kernel!(i,args...);index_body!(args...,Int32(i)-O,Val(7));return;end
function cutoff_kernel!(i,args...);cutoff_body!(args...,Int32(i)-O);return;end
function merge_kernel!(i,args...);merge_body!(args...,Int32(i)-O);return;end
function rep_kernel!(i,args...);representative_body!(args...);return;end
function make_kernel!(i,args...);table_body!(args...,Int32(i)-O,Val(false));return;end
function clean_kernel!(i,args...);table_body!(args...,Int32(i)-O,Val(true));return;end
function magic_kernel!(i,args...);magic_body!(args...,Int32(i)-O);return;end
function filter_kernel!(i,threshold,k,lengths,offsets,index,orders,words,cutoff,cluster,table,n)
 result=CUDA.CuStaticSharedArray(Int32,128);group=Int32((i-1)÷128);lane=Int32((i-1)%128)
 active=filter_sum!(offsets,index,orders,words,cluster,table,n,result,group,lane)
 if active;JACC.sync_workgroup();filter_finish!(cutoff,cluster,result,group,lane);end
 return
end
function align_kernel!(i,args...);align_body!(args...,Int32(i)-O);return;end
function jlaunch(f::F,threads,blocks,args...) where {F}
 spec=JACC.launch_spec(threads=Int(threads),blocks=Int(blocks),shmem_size=0,sync=false)
 JACC.parallel_for(spec,Int(threads)*Int(blocks),f,args...)
 nothing
end
launch_base(c)=jlaunch(base_kernel!,128,128,c.reads.data,Int64(c.total))
launch_compress(c)=jlaunch(compress_kernel!,128,cld(c.n,128),c.lengths.data,c.offsets.data,c.reads.data,c.compressed.data,c.gaps.data,c.n)
function launch_index(c)
 args=(c.reads.data,c.lengths.data,c.offsets.data,c.index.data,c.orders.data,c.words.data,c.magic.data,c.n)
 if c.k==4;jlaunch(index4_kernel!,128,cld(c.n,128),args...)
 elseif c.k==5;jlaunch(index5_kernel!,128,cld(c.n,128),args...)
 elseif c.k==6;jlaunch(index6_kernel!,128,cld(c.n,128),args...)
 else;jlaunch(index7_kernel!,128,cld(c.n,128),args...);end
 nothing
end
launch_cutoff(c)=jlaunch(cutoff_kernel!,128,cld(c.n,128),c.threshold,c.k,c.lengths.data,c.words.data,c.cutoff.data,c.n)
launch_merge(c)=jlaunch(merge_kernel!,128,cld(c.n,128),c.offsets.data,c.index.data,c.orders.data,c.words.data,c.n)
launch_rep(c)=jlaunch(rep_kernel!,1,1,c.cluster.data,c.rep.data,c.n)
launch_make(c,r)=jlaunch(make_kernel!,128,128,c.offsets.data,c.index.data,c.orders.data,c.words.data,c.table.data,r)
launch_clean(c,r)=jlaunch(clean_kernel!,128,128,c.offsets.data,c.index.data,c.orders.data,c.words.data,c.table.data,r)
launch_magic(c,r)=jlaunch(magic_kernel!,128,cld(c.n,128),c.threshold,c.lengths.data,c.magic.data,c.cluster.data,r,c.n)
launch_filter(c)=jlaunch(filter_kernel!,128,c.n,c.threshold,c.k,c.lengths.data,c.offsets.data,c.index.data,c.orders.data,c.words.data,c.cutoff.data,c.cluster.data,c.table.data,c.n)
launch_align(c,r)=jlaunch(align_kernel!,128,cld(c.n,128),c.threshold,c.lengths.data,c.offsets.data,c.compressed.data,c.gaps.data,r,c.cluster.data,c.n)
include("../study_common.jl")
main()
