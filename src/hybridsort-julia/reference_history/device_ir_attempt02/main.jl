
using CUDA
include(joinpath(@__DIR__,"..","study_device.jl"))
function device_kernel!(stage,input,output,indices,prefix,offsets,pivots,starts,finals,nulls,n,lo,hi,tpd,nr,size)
 lid=Int(threadIdx().x)-1;bx=Int(blockIdx().x)-1;ls=Int(blockDim().x);gid=bx*ls+lid;gs=Int(gridDim().x)*ls
 sh=CuStaticSharedArray(UInt32,stage isa Val{1} ? 3072 : 1024)
 bid=stage isa Val{7} ? Int(blockIdx().y)-1 : bx
 stage!(stage,gid,lid,bid,ls,gs,sh,input,output,indices,prefix,offsets,pivots,starts,finals,nulls,n,lo,hi,tpd,nr,size)
 return
end
function launch(stage,threads,blocks,args...)
 @cuda threads=threads blocks=blocks device_kernel!(stage,args...)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
