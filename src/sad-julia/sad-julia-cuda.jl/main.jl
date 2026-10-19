using CUDA
const MODEL="cuda_jl"
include("../device_common.jl")
allocate_model(a)=CuArray(a)
function compute_kernel!(sad,image,kernel,w,h,kw,kh)
    col=(Int32(blockIdx().x)-Int32(1))*Int32(16)+Int32(threadIdx().x)-Int32(1)
    row=(Int32(blockIdx().y)-Int32(1))*Int32(16)+Int32(threadIdx().y)-Int32(1)
    compute_pixel!(sad,image,kernel,w,h,kw,kh,col,row)
    return nothing
end
function minimum_kernel!(sad,minimum,n)
    tx=Int32(threadIdx().x);gid=(Int32(blockIdx().x)-Int32(1))*Int32(256)+tx-Int32(1)
    stride=Int32(gridDim().x)*Int32(256)
    cache=@cuStaticSharedMem(Int32,256)
    @inbounds cache[tx]=local_minimum(sad,n,gid,stride)
    sync_threads();i=Int32(128)
    while i!=0
        if tx<=i;@inbounds cache[tx]=min(cache[tx],cache[tx+i]);end
        sync_threads();i÷=Int32(2)
    end
    if tx==1;@inbounds CUDA.atomic_min!(pointer(minimum,17),cache[1]);end
    return nothing
end
function count_kernel!(sad,minimum,count,n)
    tx=Int32(threadIdx().x);gid=(Int32(blockIdx().x)-Int32(1))*Int32(256)+tx-Int32(1)
    s=@cuStaticSharedMem(Int32,1)
    if gid<n
        if tx==1;@inbounds s[1]=Int32(0);end
        sync_threads()
        if @inbounds sad[gid+17]==minimum[17]
            CUDA.atomic_add!(pointer(s,1),Int32(1))
        end
        sync_threads()
        if tx==1;@inbounds CUDA.atomic_add!(pointer(count,17),s[1]);end
    end
    return nothing
end
function launch_compute(sad,image,kernel,w,h,kw,kh)
    @cuda threads=(16,16) blocks=(cld(Int(w),16),cld(Int(h),16)) compute_kernel!(sad,image,kernel,w,h,kw,kh)
    nothing
end
function launch_minimum(sad,minimum,n)
    @cuda threads=256 blocks=cld(Int(n),256) minimum_kernel!(sad,minimum,n)
    nothing
end
function launch_count(sad,minimum,count,n)
    @cuda threads=256 blocks=cld(Int(n),256) count_kernel!(sad,minimum,count,n)
    nothing
end
include("../study_common.jl")
main()
