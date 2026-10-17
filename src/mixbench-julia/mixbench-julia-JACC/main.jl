using CUDA,JACC
function mixbench_jacc!(i,data,ci,blocks)
    lane=(i-1)%256;block=(i-1)÷256
    idx=block*256*8+lane+1;stride=256;big=blocks*256*8
    for k in 0:3
        @inbounds begin
            t1=data[idx+k*big];t2=data[idx+stride+k*big];t3=data[idx+2stride+k*big];t4=data[idx+3stride+k*big]
            t5=data[idx+4stride+k*big];t6=data[idx+5stride+k*big];t7=data[idx+6stride+k*big];t8=data[idx+7stride+k*big]
        end
        for _ in 1:ci
            t1=t1*t1+0.1f0;t2=t2*t2+0.1f0;t3=t3*t3+0.1f0;t4=t4*t4+0.1f0
            t5=t5*t5+0.1f0;t6=t6*t6+0.1f0;t7=t7*t7+0.1f0;t8=t8*t8+0.1f0
        end
        @inbounds data[idx+k*big]=t1*t2+t3*t4+t5*t6+t7*t8
    end
    return nothing
end
const SPEC=JACC.launch_spec(threads=256,shmem_size=0,sync=false)
function launch_jacc!(a,ci,blocks)
    JACC.parallel_for(SPEC,blocks*256,mixbench_jacc!,a,ci,Int32(blocks))
end
include(joinpath(@__DIR__,"..","study_common.jl"));mixbench_study(launch_jacc!)
