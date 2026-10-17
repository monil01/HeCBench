using CUDA,KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function bezier_ka!(inp,outp,ni,nj,resolution_i,resolution_j)
    i=(@index(Global,Linear))-1
    if i<resolution_i
    mui = Float32(i) / Float32(resolution_i - Int32(1))
    for j in Int32(0):(resolution_j - Int32(1))
        muj = Float32(j) / Float32(resolution_j - Int32(1))
        ox = Float32(0)
        oy = Float32(0)
        oz = Float32(0)
        for ki in Int32(0):ni
            bi = bezier_blend(Int(ki), mui, Int(ni))
            for kj in Int32(0):nj
                bj = bezier_blend(Int(kj), muj, Int(nj))
                p = @inbounds inp[ki * (nj + Int32(1)) + kj + Int32(1)]
                w = bi * bj
                ox += p.x * w
                oy += p.y * w
                oz += p.z * w
            end
        end
        @inbounds outp[i * resolution_j + j + Int32(1)] = XYZ(ox, oy, oz)
    end
    end
end
bezier_study((input,output,n)->()->bezier_ka!(CUDABackend(),256)(input,output,Int32(3),Int32(3),n,n;ndrange=Int(n)))
