using CUDA
include("../study_device.jl")
@inline gid()=(Int32(CUDA.blockIdx().x)-O)*Int32(CUDA.blockDim().x)+Int32(CUDA.threadIdx().x)-O
function base_kernel!(args...);base_body!(args...,gid());return;end
function compress_kernel!(args...);compress_body!(args...,gid());return;end
function index4_kernel!(args...);index_body!(args...,gid(),Val(4));return;end
function index5_kernel!(args...);index_body!(args...,gid(),Val(5));return;end
function index6_kernel!(args...);index_body!(args...,gid(),Val(6));return;end
function index7_kernel!(args...);index_body!(args...,gid(),Val(7));return;end
function cutoff_kernel!(args...);cutoff_body!(args...,gid());return;end
function merge_kernel!(args...);merge_body!(args...,gid());return;end
function rep_kernel!(args...);representative_body!(args...);return;end
function make_kernel!(args...);table_body!(args...,gid(),Val(false));return;end
function clean_kernel!(args...);table_body!(args...,gid(),Val(true));return;end
function magic_kernel!(args...);magic_body!(args...,gid());return;end
function filter_kernel!(threshold,k,lengths,offsets,index,orders,words,cutoff,cluster,table,n)
 result=CUDA.CuStaticSharedArray(Int32,128);group=Int32(CUDA.blockIdx().x)-O;lane=Int32(CUDA.threadIdx().x)-O
 active=filter_sum!(offsets,index,orders,words,cluster,table,n,result,group,lane)
 if active;CUDA.sync_threads();filter_finish!(cutoff,cluster,result,group,lane);end
 return
end
function align_kernel!(args...);align_body!(args...,gid());return;end
launch_base(c)=CUDA.@cuda threads=128 blocks=128 base_kernel!(c.reads.data,Int64(c.total))
launch_compress(c)=CUDA.@cuda threads=128 blocks=cld(c.n,128) compress_kernel!(c.lengths.data,c.offsets.data,c.reads.data,c.compressed.data,c.gaps.data,c.n)
function launch_index(c)
 args=(c.reads.data,c.lengths.data,c.offsets.data,c.index.data,c.orders.data,c.words.data,c.magic.data,c.n)
 if c.k==4;CUDA.@cuda threads=128 blocks=cld(c.n,128) index4_kernel!(args...)
 elseif c.k==5;CUDA.@cuda threads=128 blocks=cld(c.n,128) index5_kernel!(args...)
 elseif c.k==6;CUDA.@cuda threads=128 blocks=cld(c.n,128) index6_kernel!(args...)
 else;CUDA.@cuda threads=128 blocks=cld(c.n,128) index7_kernel!(args...);end
 nothing
end
launch_cutoff(c)=CUDA.@cuda threads=128 blocks=cld(c.n,128) cutoff_kernel!(c.threshold,c.k,c.lengths.data,c.words.data,c.cutoff.data,c.n)
launch_merge(c)=CUDA.@cuda threads=128 blocks=cld(c.n,128) merge_kernel!(c.offsets.data,c.index.data,c.orders.data,c.words.data,c.n)
launch_rep(c)=CUDA.@cuda threads=1 blocks=1 rep_kernel!(c.cluster.data,c.rep.data,c.n)
launch_make(c,r)=CUDA.@cuda threads=128 blocks=128 make_kernel!(c.offsets.data,c.index.data,c.orders.data,c.words.data,c.table.data,r)
launch_clean(c,r)=CUDA.@cuda threads=128 blocks=128 clean_kernel!(c.offsets.data,c.index.data,c.orders.data,c.words.data,c.table.data,r)
launch_magic(c,r)=CUDA.@cuda threads=128 blocks=cld(c.n,128) magic_kernel!(c.threshold,c.lengths.data,c.magic.data,c.cluster.data,r,c.n)
launch_filter(c)=CUDA.@cuda threads=128 blocks=c.n filter_kernel!(c.threshold,c.k,c.lengths.data,c.offsets.data,c.index.data,c.orders.data,c.words.data,c.cutoff.data,c.cluster.data,c.table.data,c.n)
launch_align(c,r)=CUDA.@cuda threads=128 blocks=cld(c.n,128) align_kernel!(c.threshold,c.lengths.data,c.offsets.data,c.compressed.data,c.gaps.data,r,c.cluster.data,c.n)
include("../study_common.jl")
main()
