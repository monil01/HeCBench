using CUDA
using Printf

const NUM_SIZE = 16

function setup_sizes()
    return [UInt(1) << (i + 5) for i in 1:NUM_SIZE]
end

function main(args)
    if length(args) != 1
        println(stderr, "Usage: main.jl <repeat>")
        return 1
    end
    repeat = parse(Int, args[1])

    for bytes in setup_sizes()
        len = Int(bytes ÷ sizeof(Int32))
        host = fill(Int32(1), len)
        dev = CuArray{Int32}(undef, len)

        for _ in 1:repeat
            copyto!(dev, host)
        end
        CUDA.synchronize()

        start = time_ns()
        for _ in 1:repeat
            copyto!(dev, host)
        end
        CUDA.synchronize()
        time_h2d = Int64(time_ns() - start)
        @printf("Copy %d bytes from host to device takes %f us\n",
                bytes, (time_h2d * 1.0e-3) / repeat)

        for _ in 1:repeat
            copyto!(host, dev)
        end
        CUDA.synchronize()

        start = time_ns()
        for _ in 1:repeat
            copyto!(host, dev)
        end
        CUDA.synchronize()
        time_d2h = Int64(time_ns() - start)
        @printf("Copy %d bytes from device to host takes %f us\n",
                bytes, (time_d2h * 1.0e-3) / repeat)
        @printf("Timing gap in nanoseconds per byte: %f\n\n",
                abs(time_h2d - time_d2h) / Float64(repeat * Int(bytes)))
    end

    return 0
end

exit(main(ARGS))
