using CUDA,KernelAbstractions
import Atomix
const MODEL="ka"
include("../device_common.jl")
function allocate_model(a)
    out=KernelAbstractions.allocate(CUDABackend(),eltype(a),size(a));copyto!(out,a);out
end
@kernel function compute_kernel!(sad,image,kernel,w,h,kw,kh)
    global_id=@index(Global,Cartesian)
    col=Int32(global_id[1])-Int32(1);row=Int32(global_id[2])-Int32(1)
    compute_pixel!(sad,image,kernel,w,h,kw,kh,col,row)
end
@kernel function minimum_kernel!(sad,minimum,n,stride)
    local_id=@index(Local,Linear)
    global_id=@index(Global,Linear)
    tx=Int32(local_id);gid=Int32(global_id)-Int32(1)
    cache=@localmem Int32 (256,)
    @inbounds cache[tx]=local_minimum(sad,n,gid,stride)
    @synchronize
    i=Int32(128)
    while i!=0
        if tx<=i
            @inbounds cache[tx]=min(cache[tx],cache[tx+i])
        end
        @synchronize
        i÷=Int32(2)
    end
    if tx==1
        @inbounds Atomix.@atomic min(minimum[17],cache[1])
    end
end
@kernel function count_kernel!(sad,minimum,count,n)
    local_id=@index(Local,Linear)
    global_id=@index(Global,Linear)
    tx=Int32(local_id);gid=Int32(global_id)-Int32(1)
    s=@localmem Int32 (1,)
    if tx==1
        @inbounds s[1]=Int32(0)
    end
    @synchronize
    if gid<n && (@inbounds sad[gid+17]==minimum[17])
        @inbounds Atomix.@atomic s[1]+=Int32(1)
    end
    @synchronize
    if tx==1
        @inbounds Atomix.@atomic count[17]+=s[1]
    end
end
function launch_compute(sad,image,kernel,w,h,kw,kh)
    compute_kernel!(CUDABackend(),(16,16))(sad,image,kernel,w,h,kw,kh;ndrange=(cld(Int(w),16)*16,cld(Int(h),16)*16))
    nothing
end
function launch_minimum(sad,minimum,n)
    padded=cld(Int(n),256)*256
    minimum_kernel!(CUDABackend(),256)(sad,minimum,n,Int32(padded);ndrange=padded)
    nothing
end
function launch_count(sad,minimum,count,n)
    count_kernel!(CUDABackend(),256)(sad,minimum,count,n;ndrange=cld(Int(n),256)*256)
    nothing
end
include("../study_common.jl")
main()
