using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","device_components.jl"))
@kernel function compress_kernel!(n,original,compressed,goffset,dim)
    S=@localmem UInt32 (4098,)
    compress_body!(S,n,original,compressed,goffset,dim)
    nothing
end
@kernel function decompress_kernel!(compressed,decompressed,goffset)
    S=@localmem UInt32 (4098,)
    decompress_body!(S,compressed,decompressed,goffset)
    nothing
end
function backend_array(x)
    a=KernelAbstractions.allocate(CUDABackend(),eltype(x),(length(x),));copyto!(a,x);a
end
function build_kernels(n,I,E,D,O,dim)
    # Explicit CUDA backend adapter preserves original launch_bounds(1024,2),
    # needed by this source inter-block resident-ring forward-progress protocol.
    co=compress_kernel!(CUDABackend(),1024);de=decompress_kernel!(CUDABackend(),1024)
    nd,wg,it,dy=KernelAbstractions.launch_config(co,160*1024,1024);ctxc=KernelAbstractions.mkcontext(co,nd,it)
    nd,wg,it,dy=KernelAbstractions.launch_config(de,160*1024,1024);ctxd=KernelAbstractions.mkcontext(de,nd,it)
    c=CUDA.@cuda launch=false maxthreads=1024 blocks_per_sm=2 co.f(ctxc,n,I,E,O,dim)
    d=CUDA.@cuda launch=false maxthreads=1024 blocks_per_sm=2 de.f(ctxd,E,D,O)
    for (name,k) in (("compress",c),("decompress",d))
        active=CUDA.active_blocks(k.fun,1024)
        println("STUDY_KERNEL_RESOURCE model=ka kernel=$name registers=$(CUDA.registers(k)) memory=$(CUDA.memory(k)) active_blocks_per_SM=$active maxthreads=1024 blocks_per_sm=2")
        active>=2||error("sourceinterblockringrequires2residentblocks/SM; no unsafe launch")
    end
    (n,I,E,O,dim)->c(ctxc,n,I,E,O,dim;threads=1024,blocks=160), (E,D,O)->d(ctxd,E,D,O;threads=1024,blocks=160)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
