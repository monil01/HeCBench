using CUDA
import JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","device_components.jl"))
function compress_kernel!(index,n,original,compressed,goffset,dim,template)
    S=JACC.shared(template)
    compress_body!(S,n,original,compressed,goffset,dim)
    nothing
end
function decompress_kernel!(index,compressed,decompressed,goffset,template)
    S=JACC.shared(template)
    decompress_body!(S,compressed,decompressed,goffset)
    nothing
end
backend_array(x)=JACC.array(x)
const TEMPLATE=JACC.array(zeros(UInt32,4098))
all(iszero,Array(TEMPLATE))||error("actualGPUtemplateidentity")
println("STUDY_BACKEND_INPUT actual_GPU_zero_UInt32_shared_template4098_checked=true; allalgorithmically_read_sharedslots_overwritten_beforeuse=true")
const JACC_CUDA_EXTENSION=Base.get_extension(JACC,:CUDAExt)
function build_kernels(n,I,E,D,O,dim)
    # Pinned LaunchSpec does not expose cufunction launch-bound keywords.
    # Compile the actual JACC generated wrapper, with the exact source
    # launch_bounds(1024,2), to preserve this resident-ring progress contract.
    ex=JACC_CUDA_EXTENSION;N=160*1024
    ca=ex._kernel_args(N,compress_kernel!,n,I,E,O,dim,TEMPLATE)
    da=ex._kernel_args(N,decompress_kernel!,E,D,O,TEMPLATE)
    c=CUDA.cufunction(ex._parallel_for_cuda,Tuple{Core.Typeof.(ca)...};maxthreads=1024,blocks_per_sm=2)
    d=CUDA.cufunction(ex._parallel_for_cuda,Tuple{Core.Typeof.(da)...};maxthreads=1024,blocks_per_sm=2)
    for (name,k) in (("compress",c),("decompress",d))
        active=CUDA.active_blocks(k.fun,1024;shmem=16392)
        println("STUDY_KERNEL_RESOURCE model=jacc kernel=$name registers=$(CUDA.registers(k)) memory=$(CUDA.memory(k)) dynamic_shared=16392 active_blocks_per_SM=$active maxthreads=1024 blocks_per_sm=2")
        active>=2||error("sourceinterblockringrequires2residentblocks/SM; no unsafe launch")
    end
    (n,I,E,O,dim)->c(N,compress_kernel!,n,I,E,O,dim,TEMPLATE;threads=1024,blocks=160,shmem=16392), (E,D,O)->d(N,decompress_kernel!,E,D,O,TEMPLATE;threads=1024,blocks=160,shmem=16392)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
main()
