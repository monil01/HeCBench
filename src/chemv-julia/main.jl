using CUDA
using Printf

const REPEAT = 1000
const N = 370
const LDAT = N
const INCX = 1
const INCY = 1
const AT_SIZE = N * LDAT
const X_SIZE = N * INCX
const Y_SIZE = N * INCY

struct ComplexFloat
    Re::Float32
    Im::Float32
end

function chemv_cpu!(alpha_re::Float32, alpha_im::Float32, beta_re::Float32, beta_im::Float32,
                    at::Vector{ComplexFloat}, x::Vector{ComplexFloat}, y::Vector{ComplexFloat})
    @inbounds for i0 in 0:N-1
        yi = y[i0 * INCY + 1]
        y[i0 * INCY + 1] = ComplexFloat(yi.Re * beta_re - yi.Im * beta_im,
                                        yi.Im * beta_re + yi.Re * beta_im)
    end

    @inbounds for i1 in 0:N-1
        a = at[i1 * LDAT + i1 + 1]
        xi = x[i1 * INCX + 1]
        yi = y[i1 * INCY + 1]
        var2_re = alpha_re * a.Re
        var2_im = alpha_im * a.Re
        var3_re = var2_re * xi.Re - var2_im * xi.Im
        var3_im = var2_im * xi.Re + var2_re * xi.Im
        y[i1 * INCY + 1] = ComplexFloat(yi.Re + var3_re, yi.Im + var3_im)
    end

    @inbounds for i2 in 0:N-2
        for i3 in 0:N-2-i2
            a = at[i2 * LDAT + (1 + i2) + i3 + 1]
            xhi = x[(i3 + i2 + 1) * INCX + 1]
            yi = y[i2 * INCY + 1]
            var94_re = alpha_re * a.Re - alpha_im * (-a.Im)
            var94_im = alpha_im * a.Re + alpha_re * (-a.Im)
            var95_re = var94_re * xhi.Re - var94_im * xhi.Im
            var95_im = var94_im * xhi.Re + var94_re * xhi.Im
            y[i2 * INCY + 1] = ComplexFloat(yi.Re + var95_re, yi.Im + var95_im)

            xlo = x[i2 * INCX + 1]
            yj = y[(i3 + i2 + 1) * INCY + 1]
            var97_re = alpha_re * a.Re - alpha_im * a.Im
            var97_im = alpha_im * a.Re + alpha_re * a.Im
            var98_re = var97_re * xlo.Re - var97_im * xlo.Im
            var98_im = var97_im * xlo.Re + var97_re * xlo.Im
            y[(i3 + i2 + 1) * INCY + 1] = ComplexFloat(yj.Re + var98_re, yj.Im + var98_im)
        end
    end
    return
end

function chemv_kernel0!(at, x, y, alpha_im::Float32, alpha_re::Float32, beta_im::Float32, beta_re::Float32)
    b0 = blockIdx().x - Int32(1)
    t0 = threadIdx().x - Int32(1)
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
        sync_threads()
    end
    return
end

function chemv_kernel1!(at, x, y, alpha_im::Float32, alpha_re::Float32)
    b0 = blockIdx().x - Int32(1)
    t0 = threadIdx().x - Int32(1)
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
        sync_threads()
    end
    return
end

function chemv_gpu!(alpha_re::Float32, alpha_im::Float32, beta_re::Float32, beta_im::Float32,
                    at::Vector{ComplexFloat}, x::Vector{ComplexFloat}, y::Vector{ComplexFloat})
    d_at = CuArray(at)
    d_x = CuArray(x)
    d_y = CuArray(y)

    CUDA.synchronize()
    start = time_ns()
    for _ in 1:REPEAT
        @cuda threads=32 blocks=12 chemv_kernel0!(d_at, d_x, d_y, alpha_im, alpha_re, beta_im, beta_re)
        @cuda threads=32 blocks=12 chemv_kernel1!(d_at, d_x, d_y, alpha_im, alpha_re)
    end
    CUDA.synchronize()
    elapsed = time_ns() - start
    @printf("Average execution time of chemv kernels: %f (us)\n", elapsed * 1.0e-3 / REPEAT)

    y .= Array(d_y)
    return
end

function main()
    at = Vector{ComplexFloat}(undef, AT_SIZE)
    x = Vector{ComplexFloat}(undef, X_SIZE)
    y_cpu = Vector{ComplexFloat}(undef, Y_SIZE)
    y_gpu = Vector{ComplexFloat}(undef, Y_SIZE)

    @inbounds for i in 0:N-1
        x[i * INCX + 1] = ComplexFloat(Float32(i + 5), Float32(i * 2))
        y_cpu[i * INCY + 1] = ComplexFloat(Float32(i * 3), Float32(i + 7))
        y_gpu[i * INCY + 1] = ComplexFloat(Float32(i * 3), Float32(i + 7))
        for j in 0:LDAT-1
            at[i * LDAT + j + 1] = ComplexFloat(Float32(i + j), Float32(i + 3))
        end
    end

    alpha_re = Float32(3.14)
    alpha_im = Float32(1.59)
    beta_re = Float32(2.71)
    beta_im = Float32(8.28)

    chemv_cpu!(alpha_re, alpha_im, beta_re, beta_im, at, x, y_cpu)
    chemv_gpu!(alpha_re, alpha_im, beta_re, beta_im, at, x, y_gpu)

    @inbounds for i in 0:N-1
        yc = y_cpu[i * INCY + 1]
        yg = y_gpu[i * INCY + 1]
        if abs(yc.Re - yg.Re) > 1.0f-3 || abs(yc.Im - yg.Im) > 1.0f-3
            @printf("%d %f %f\n", i, yc.Re, yg.Re)
            println("FAIL")
            return 1
        end
    end
    println("PASS")
    return 0
end

exit(main())
