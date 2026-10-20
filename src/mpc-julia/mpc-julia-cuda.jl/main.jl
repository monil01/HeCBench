using CUDA
include(joinpath(@__DIR__,"..","device_components.jl"))
function compress_kernel!(n,original,compressed,goffset,dim)
    S=CUDA.CuStaticSharedArray(UInt32,4098)
    compress_body!(S,n,original,compressed,goffset,dim)
    nothing
end
function decompress_kernel!(compressed,decompressed,goffset)
    S=CUDA.CuStaticSharedArray(UInt32,4098)
    decompress_body!(S,compressed,decompressed,goffset)
    nothing
end
backend_array(x)=CuArray(x)
function build_kernels(n,I,E,D,O,dim)
    c=CUDA.@cuda launch=false maxthreads=1024 blocks_per_sm=2 compress_kernel!(n,I,E,O,dim)
    d=CUDA.@cuda launch=false maxthreads=1024 blocks_per_sm=2 decompress_kernel!(E,D,O)
    for (name,k) in (("compress",c),("decompress",d))
        active=CUDA.active_blocks(k.fun,1024)
        println("STUDY_KERNEL_RESOURCE model=cuda_jl kernel=$name registers=$(CUDA.registers(k)) memory=$(CUDA.memory(k)) active_blocks_per_SM=$active maxthreads=1024 blocks_per_sm=2")
        active>=2||error("sourceinterblockringrequires2residentblocks/SM; no unsafe launch")
    end
    (n,I,E,O,dim)->c(n,I,E,O,dim;threads=1024,blocks=160), (E,D,O)->d(E,D,O;threads=1024,blocks=160)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
