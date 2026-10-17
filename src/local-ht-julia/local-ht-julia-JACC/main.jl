using CUDA,JACC
JACC.@init_backend
include(joinpath(@__DIR__,"..","study_kernel_body.jl"))
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function assembly_jacc!(i,args...)
 index=Int32(i)-Int32(1)
 assembly_body!(index÷Int32(32),mod(index,Int32(32)),args...)
 nothing
end
@inline function capture_jacc!(i,keylen,keypos,reads,hashes,capacity,total)
 capture_hash_body!(Int32(i)-Int32(1),keylen,keypos,reads,hashes,capacity,total)
 nothing
end
const MAIN_SPEC=JACC.launch_spec(threads=512,blocks=1,shmem_size=0,sync=false)
const CAPTURE_SPEC=JACC.launch_spec(threads=256,blocks=10,shmem_size=0,sync=false)
const JACC_CUDA_EXTENSION=Base.get_extension(JACC,:CUDAExt)
function run_main!(args...)
 # JACC1.4.0 exposes no maxregs compiler option in LaunchSpec. Compile its
 # actual parallel_for wrapper with an explicit cap; the device algorithm
 # remains assembly_jacc! through JACC's own generated indexing wrapper.
 kargs=JACC_CUDA_EXTENSION._kernel_args(512,assembly_jacc!,args...)
 kernel=CUDA.cufunction(JACC_CUDA_EXTENSION._parallel_for_cuda,Tuple{Core.Typeof.(kargs)...};maxregs=128)
 kernel(kargs...;threads=512,blocks=1,shmem=0,stream=CUDA.stream())
 nothing
end
function run_hash!(args...)
 JACC.parallel_for(CAPTURE_SPEC,2560,capture_jacc!,args...)
 nothing
end
study(run_main!,run_hash!)
