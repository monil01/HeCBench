using CUDA
using KernelAbstractions
include(joinpath(@__DIR__,"..","study_common.jl"))
@kernel function portable_mrc1!(n::Int32,y,x1,x2,dout,margin::Float32,dx1,dx2)
    i=Int32(@index(Global,Linear))
    if i <= n
        @inbounds begin
            yf = Float32(y[i])
            dist = -yf * (x1[i] - x2[i]) + margin
            if dist < 0.0f0
                dx1[i] = 0.0f0
                dx2[i] = 0.0f0
            else
                dx1[i] = -yf * dout[i]
                dx2[i] = yf * dout[i]
            end
        end
    end
    nothing
end
@kernel function portable_mrc2!(n::Int32,y,x1,x2,dout,margin::Float32,dx1,dx2)
    i=Int32(@index(Global,Linear))
    if i <= n
        @inbounds begin
            yf = Float32(y[i])
            o = dout[i]
            dist = -yf * (x1[i] - x2[i]) + margin
            dx1[i] = dist < 0.0f0 ? 0.0f0 : -yf * o
            dx2[i] = dist < 0.0f0 ? 0.0f0 : yf * o
        end
    end
    nothing
end
@kernel function portable_mrc3!(n::Int32,y,x1,x2,dout,margin::Float32,dx1,dx2)
    i=Int32(@index(Global,Linear))
    stride=n÷Int32(4)
    while i <= n
        @inbounds begin
            yf = Float32(y[i])
            o = dout[i]
            dist = -yf * (x1[i] - x2[i]) + margin
            dx1[i] = dist < 0.0f0 ? 0.0f0 : -yf * o
            dx2[i] = dist < 0.0f0 ? 0.0f0 : yf * o
        end
        i += stride
    end
    nothing
end
function study_prepare(y,a,b,d,o1,o2,n,m)
    backend=CUDABackend();k1=portable_mrc1!(backend,256);k2=portable_mrc2!(backend,256);k3=portable_mrc3!(backend,256)
    function launch!(mode)
        if mode==1
            k1(Int32(n),y,a,b,d,m,o1,o2;ndrange=n)
        elseif mode==2
            k2(Int32(n),y,a,b,d,m,o1,o2;ndrange=n)
        else
            k3(Int32(n),y,a,b,d,m,o1,o2;ndrange=n÷4)
        end
    end
end
exit(study_main(study_prepare))
