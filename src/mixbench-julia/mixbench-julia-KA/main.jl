using CUDA,KernelAbstractions
@kernel function mixbench_ka!(data,ci,blocks)
    lane=@index(Local,Linear)-1;block=@index(Group,Linear)-1
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
end
function launch_ka!(a,ci,blocks)
    mixbench_ka!(get_backend(a),256)(a,ci,Int32(blocks);ndrange=blocks*256)
end
include(joinpath(@__DIR__,"..","study_common.jl"));mixbench_study(launch_ka!)
