using CUDA
import JACC
JACC.@init_backend
const MODEL="jacc"
include("../device_common.jl")
allocate_model(a)=JACC.array(a)
const MIN_TEMPLATE=JACC.array(zeros(Int32,256))
const COUNT_TEMPLATE=JACC.array(zeros(Int32,1))
function audit_templates(label)
    m=Array(MIN_TEMPLATE);c=Array(COUNT_TEMPLATE)
    @assert length(m)==256 && all(iszero,m) && c==Int32[0]
    println("STUDY_SHARED_TEMPLATE_",label," min=",repr(m)," count=",repr(c));flush(stdout)
end
audit_templates("INITIAL")
function compute_kernel!(i,j,sad,image,kernel,w,h,kw,kh)
    compute_pixel!(sad,image,kernel,w,h,kw,kh,Int32(i)-Int32(1),Int32(j)-Int32(1))
    return nothing
end
function minimum_kernel!(i,sad,minimum,n,stride,template)
    id=Int32(i)-Int32(1);tx=id%Int32(256)+Int32(1)
    cache=JACC.shared(template)
    @inbounds cache[tx]=local_minimum(sad,n,id,stride)
    JACC.sync_workgroup();offset=Int32(128)
    while offset!=0
        if tx<=offset
            @inbounds cache[tx]=min(cache[tx],cache[tx+offset])
        end
        JACC.sync_workgroup();offset÷=Int32(2)
    end
    if tx==1
        @inbounds JACC.@atomic min(minimum[17],cache[1])
    end
    return nothing
end
function count_kernel!(i,sad,minimum,count,n,template)
    id=Int32(i)-Int32(1);tx=id%Int32(256)+Int32(1)
    s=JACC.shared(template)
    if tx==1
        @inbounds s[1]=Int32(0)
    end
    JACC.sync_workgroup()
    if id<n && (@inbounds sad[id+17]==minimum[17])
        @inbounds JACC.@atomic s[1]+=Int32(1)
    end
    JACC.sync_workgroup()
    if tx==1
        @inbounds JACC.@atomic count[17]+=s[1]
    end
    return nothing
end
function launch_compute(sad,image,kernel,w,h,kw,kh)
    blocks=(cld(Int(w),16),cld(Int(h),16));extent=(blocks[1]*16,blocks[2]*16)
    spec=JACC.launch_spec(threads=(16,16),blocks=blocks,shmem_size=0,sync=false)
    JACC.parallel_for(spec,extent,compute_kernel!,sad,image,kernel,w,h,kw,kh)
    nothing
end
function launch_minimum(sad,minimum,n)
    blocks=cld(Int(n),256);padded=blocks*256
    spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=1024,sync=false)
    JACC.parallel_for(spec,padded,minimum_kernel!,sad,minimum,n,Int32(padded),MIN_TEMPLATE)
    nothing
end
function launch_count(sad,minimum,count,n)
    blocks=cld(Int(n),256);padded=blocks*256
    spec=JACC.launch_spec(threads=256,blocks=blocks,shmem_size=4,sync=false)
    JACC.parallel_for(spec,padded,count_kernel!,sad,minimum,count,n,COUNT_TEMPLATE)
    nothing
end
include("../study_common.jl")
main()
audit_templates("FINAL")
