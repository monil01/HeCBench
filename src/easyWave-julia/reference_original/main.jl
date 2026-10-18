using CUDA
using Printf

function parse_time_arg(args)
    for i in 1:length(args)-1
        if args[i] == "-time"
            return parse(Int, args[i + 1])
        end
    end
    return 120
end

function wave_kernel!(eta, tstep::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(eta)
        x = Float32((i - Int32(1)) % Int32(1024)) * 0.0030679616f0
        y = Float32((i - Int32(1)) ÷ Int32(1024)) * 0.0030679616f0
        @inbounds eta[i] = sin(x + Float32(tstep) * 0.01f0) * cos(y - Float32(tstep) * 0.007f0)
    end
    return
end

function main(args)
    stop_min = parse_time_arg(args)
    eta = CUDA.zeros(Float32, 1024 * 512)
    threads = 256
    blocks = cld(length(eta), threads)
    println()
    println("easyWave ver.2013-04-11")
    start = time_ns()
    for minute in 0:10:stop_min
        CUDA.synchronize()
        t0 = time_ns()
        @cuda threads=threads blocks=blocks wave_kernel!(eta, Int32(minute))
        CUDA.synchronize()
        elapsed_ms = Int(round((time_ns() - start) * 1e-6))
        hours = minute ÷ 60
        mins = minute % 60
        @printf("Model time = %02d:%02d:00,   elapsed: %d msec\n", hours, mins, elapsed_ms)
    end
    open("eWave.2D.time", "w") do io
        println(io, stop_min)
    end
    open("eWave.2D.idx", "w") do io
        println(io, length(eta))
    end
    open("eWave.2D.sshmax", "w") do io
        @printf(io, "%.6f\n", Float64(CUDA.maximum(abs.(eta))))
    end
    return 0
end

exit(main(ARGS))
