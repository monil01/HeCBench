
using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_device.jl"))
@kernel function portable_kernel!(stage::Val{S},input,output,indices,prefix,offsets,pivots,starts,finals,nulls,n,lo,hi,tpd,nr,size) where {S}
 lid=@index(Local,Linear);grp=@index(Group,NTuple)
 sh=@localmem UInt32 (S==1 ? 3072 : 1024,)
 localid=lid-1;bx=grp[1]-1;ls=@groupsize()[1];gs=@ndrange()[1]
 gid=bx*ls+localid;bid=S==7 ? grp[2]-1 : bx
 stage!(stage,gid,localid,bid,ls,gs,sh,input,output,indices,prefix,offsets,pivots,starts,finals,nulls,n,lo,hi,tpd,nr,size)
end
function launch(stage,threads,blocks,args...)
 bx=blocks isa Tuple ? blocks[1] : blocks;by=blocks isa Tuple ? blocks[2] : 1
 portable_kernel!(CUDABackend(),(threads,1))(stage,args...;ndrange=(threads*bx,by))
end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
