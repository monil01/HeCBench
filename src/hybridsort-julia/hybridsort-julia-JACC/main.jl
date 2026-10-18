
using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_device.jl"))
function portable_callback!(i,j,stage::Val{S},input,output,indices,prefix,offsets,pivots,starts,finals,nulls,n,lo,hi,tpd,nr,size) where {S}
 lid=Int(CUDA.threadIdx().x)-1;bx=Int(CUDA.blockIdx().x)-1;ls=Int(CUDA.blockDim().x);gs=Int(CUDA.gridDim().x)*ls
 sh=CUDA.CuStaticSharedArray(UInt32,S==1 ? 3072 : 1024)
 bid=S==7 ? Int(CUDA.blockIdx().y)-1 : bx
 stage!(stage,Int(i)-1,lid,bid,ls,gs,sh,input,output,indices,prefix,offsets,pivots,starts,finals,nulls,n,lo,hi,tpd,nr,size)
 nothing
end
function launch(stage,threads,blocks,args...)
 bx=blocks isa Tuple ? blocks[1] : blocks;by=blocks isa Tuple ? blocks[2] : 1
 spec=JACC.launch_spec(;threads=(threads,1),blocks=(bx,by),shmem_size=0,sync=false)
 JACC.parallel_for(spec,(threads*bx,by),portable_callback!,stage,args...)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
