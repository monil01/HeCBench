using JACC
include("../study_common.jl")
# Public shared(template) uses dynamic baseoffsetzero. One Int64arena avoids
# aliasing independent shared allocations; explicitwidth/layout adaptation.
struct Segment{T}
 data::T
 offset::Int32
end
@inline Base.getindex(v::Segment,i)=@inbounds v.data[v.offset+i]
@inline Base.setindex!(v::Segment,x,i)=(@inbounds v.data[v.offset+i]=x)
@inline function kernel(i,result,v::Val{N},template) where N
 arena=JACC.shared(template);a=4*(2N+1);o=4*(4N+2);s=4*24N
 av=Segment(arena,Int32(0));op=Segment(arena,Int32(a));st=Segment(arena,Int32(a+o));ps=Segment(arena,Int32(a+o+s))
 lid=(i-1)%4+1
 dfs_lane!(result,av,op,st,ps,lid,i,v)
 return
end
const BACKEND=JACC.get_backend(:cuda)
function factory(result,v::Val{N}) where N
 count=4*((2N+1)+(4N+2)+24N+8*((N+7)÷8));template=CUDA.zeros(Int64,count)
 spec=JACC.LaunchSpec{typeof(BACKEND)}(;threads=4,blocks=4096,shmem_size=8count,sync=false)
 ()->JACC.parallel_for(spec,16384,kernel,result,v,template)
end
langford_study(factory)
