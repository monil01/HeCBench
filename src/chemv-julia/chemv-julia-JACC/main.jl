using CUDA,JACC
struct ComplexFloat
Re::Float32
Im::Float32
end
include(joinpath(@__DIR__,"..","study_common.jl"))
function chemv_kernel0!(index,at, x, y, alpha_im::Float32, alpha_re::Float32, beta_im::Float32, beta_re::Float32)
    b0 = Int32(div(index-1,32))
    t0 = Int32(mod(index-1,32))
    row = Int32(32) * b0 + t0

    for c1 in Int32(0):Int32(32):min(Int32(368), Int32(32) * b0 + Int32(30))
        if row <= Int32(369) && c1 == Int32(0)
            yi = @inbounds y[row + Int32(1)]
            yscaled = ComplexFloat(yi.Re * beta_re - yi.Im * beta_im,
                                   yi.Im * beta_re + yi.Re * beta_im)
            @inbounds y[row + Int32(1)] = yscaled
            a = @inbounds at[Int32(11872) * b0 + Int32(371) * t0 + Int32(1)]
            xi = @inbounds x[row + Int32(1)]
            var2_re = alpha_re * a.Re
            var2_im = alpha_im * a.Re
            var3_re = var2_re * xi.Re - var2_im * xi.Im
            var3_im = var2_im * xi.Re + var2_re * xi.Im
            @inbounds y[row + Int32(1)] = ComplexFloat(yscaled.Re + var3_re, yscaled.Im + var3_im)
        end
        if row <= Int32(369)
            for c3 in Int32(0):min(Int32(31), row - c1 - Int32(1))
                a = @inbounds at[row + Int32(370) * c1 + Int32(370) * c3 + Int32(1)]
                xc = @inbounds x[c1 + c3 + Int32(1)]
                yi = @inbounds y[row + Int32(1)]
                var97_re = alpha_re * a.Re - alpha_im * a.Im
                var97_im = alpha_im * a.Re + alpha_re * a.Im
                var98_re = var97_re * xc.Re - var97_im * xc.Im
                var98_im = var97_im * xc.Re + var97_re * xc.Im
                @inbounds y[row + Int32(1)] = ComplexFloat(yi.Re + var98_re, yi.Im + var98_im)
            end
        end
        JACC.sync_workgroup()
    end
    return
end

function chemv_kernel1!(index,at, x, y, alpha_im::Float32, alpha_re::Float32)
    b0 = Int32(div(index-1,32))
    t0 = Int32(mod(index-1,32))
    row = Int32(32) * b0 + t0

    for c1 in Int32(5888) * b0:Int32(32):min(Int32(67712), Int32(5856) * b0 + Int32(6016))
        lo = max(Int32(0), Int32(5888) * b0 + Int32(184) * t0 - c1)
        hi = min(Int32(31), Int32(5856) * b0 + Int32(183) * t0 - c1 + Int32(368))
        for c3 in lo:hi
            idx = Int32(5984) * b0 + Int32(187) * t0 + c1 + c3 + Int32(2)
            a = @inbounds at[idx]
            xidx = -Int32(5856) * b0 - Int32(183) * t0 + c1 + c3 + Int32(2)
            xv = @inbounds x[xidx]
            yi = @inbounds y[row + Int32(1)]
            var94_re = alpha_re * a.Re - alpha_im * (-a.Im)
            var94_im = alpha_im * a.Re + alpha_re * (-a.Im)
            var95_re = var94_re * xv.Re - var94_im * xv.Im
            var95_im = var94_im * xv.Re + var94_re * xv.Im
            @inbounds y[row + Int32(1)] = ComplexFloat(yi.Re + var95_re, yi.Im + var95_im)
        end
        JACC.sync_workgroup()
    end
    return
end

function factory(at,x,y)
 spec=JACC.launch_spec(;threads=32,shmem_size=0,sync=false)
 return ()->begin
 JACC.parallel_for(spec,384,chemv_kernel0!,at,x,y,1.59f0,3.14f0,8.28f0,2.71f0)
 JACC.parallel_for(spec,384,chemv_kernel1!,at,x,y,1.59f0,3.14f0)
 end
end
chemv_study(factory)
