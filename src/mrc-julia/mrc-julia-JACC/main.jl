using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
function portable_mrc1!(idx,n::Int32,y,x1,x2,dout,margin::Float32,dx1,dx2)
    i=Int32(idx)
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
function portable_mrc2!(idx,n::Int32,y,x1,x2,dout,margin::Float32,dx1,dx2)
    i=Int32(idx)
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
function portable_mrc3!(idx,n::Int32,y,x1,x2,dout,margin::Float32,dx1,dx2)
    i=Int32(idx)
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
    backend=JACC.get_backend(:cuda);ST=JACC.LaunchSpec{typeof(backend)};spec=ST(;threads=256,blocks=n÷256,shmem_size=0,sync=false);spec3=ST(;threads=256,blocks=n÷1024,shmem_size=0,sync=false)
    function launch!(mode)
        if mode==1
            JACC.parallel_for(portable_mrc1!,spec,n,Int32(n),y,a,b,d,m,o1,o2)
        elseif mode==2
            JACC.parallel_for(portable_mrc2!,spec,n,Int32(n),y,a,b,d,m,o1,o2)
        else
            JACC.parallel_for(portable_mrc3!,spec3,n÷4,Int32(n),y,a,b,d,m,o1,o2)
        end
    end
end
exit(study_main(study_prepare))
