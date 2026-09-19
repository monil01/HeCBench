using CUDA
using Printf

function main(args)
    if length(args) != 3
        println("Usage: ./main <minimum copy length> <maximum copy length> <repeat>")
        return 1
    end

    min_len = parse(Int, args[1])
    max_len = parse(Int, args[2])
    repeat = parse(Int, args[3])

    devs = collect(CUDA.devices())
    num_devices = max(length(devs), 1)
    for dev in devs
        CUDA.device!(dev) do
            @printf("Device name: %s\n", CUDA.name(dev))
        end
    end
    if num_devices <= 1
        println("Warning, only one device is detected.This program is supposed to execute with multiple devices.")
    end

    len = min_len
    ok_all = true
    while len <= max_len
        host = collect(Int32(0):Int32(len - 1))
        buffers = Vector{CuArray{Int32,1}}(undef, num_devices)
        for i in 1:num_devices
            if !isempty(devs)
                CUDA.device!(devs[i]) do
                    buffers[i] = CUDA.zeros(Int32, len)
                end
            else
                buffers[i] = CUDA.zeros(Int32, len)
            end
        end
        copyto!(buffers[1], host)
        fill!(host, Int32(0))

        CUDA.synchronize()
        start = time_ns()
        for _ in 1:repeat
            for i in 1:num_devices
                src = buffers[i]
                dst = buffers[mod1(i + 1, num_devices)]
                copyto!(dst, src)
            end
        end
        CUDA.synchronize()
        time_us = (time_ns() - start) * 1.0e-3 / repeat
        println("----------------------------------------------------------------")
        @printf("Copy length = %d\n", len)
        @printf("Average total exchange time: %f (us)\n", time_us)
        @printf("Average exchange time per device: %f (us)\n", time_us / num_devices)

        result = Array(buffers[1])
        ok = all(result .== collect(Int32(0):Int32(len - 1)))
        ok_all &= ok
        println(ok ? "PASS" : "FAIL")
        len *= 4
    end
    return ok_all ? 0 : 1
end

exit(main(ARGS))
