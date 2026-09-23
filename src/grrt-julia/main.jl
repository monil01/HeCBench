using CUDA
using Printf

const IMAGE_SIZE = 1024
const PI32 = Float32(3.14159265)

function task1_kernel!(results)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total = Int32(IMAGE_SIZE * IMAGE_SIZE)
    if idx <= total
        pix = idx - Int32(1)
        x = pix % Int32(IMAGE_SIZE)
        y = pix ÷ Int32(IMAGE_SIZE)
        alpha = (Float32(x) / Float32(IMAGE_SIZE - 1) - 0.5f0) * 2.0f0
        beta = (Float32(y) / Float32(IMAGE_SIZE - 1) - 0.5f0) * 2.0f0
        redshift = inv(1.0f0 + sqrt(alpha * alpha + beta * beta))
        base = Int32(3) * pix
        @inbounds results[base + Int32(1)] = Float64(alpha)
        @inbounds results[base + Int32(2)] = Float64(beta)
        @inbounds results[base + Int32(3)] = Float64(redshift)
    end
    return
end

function task2_kernel!(results)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    total = Int32(IMAGE_SIZE * IMAGE_SIZE)
    if idx <= total
        pix = idx - Int32(1)
        x = pix % Int32(IMAGE_SIZE)
        y = pix ÷ Int32(IMAGE_SIZE)
        alpha = (Float32(x) / Float32(IMAGE_SIZE - 1) - 0.5f0) * 2.0f0
        beta = (Float32(y) / Float32(IMAGE_SIZE - 1) - 0.5f0) * 2.0f0
        r2 = alpha * alpha + beta * beta
        luminosity = exp(-4.0f0 * r2) * cos(PI32 * alpha * 0.25f0)^2
        base = Int32(3) * pix
        @inbounds results[base + Int32(1)] = Float64(alpha)
        @inbounds results[base + Int32(2)] = Float64(beta)
        @inbounds results[base + Int32(3)] = Float64(luminosity)
    end
    return
end

function write_output(path, header, host)
    open(path, "w") do io
        println(io, header)
        for j in 0:IMAGE_SIZE-1
            row = IMAGE_SIZE * j
            for i in 0:IMAGE_SIZE-1
                base = 3 * (row + i)
                @printf(io, "%f\t%f\t%f\n", Float32(host[base + 1]), Float32(host[base + 2]), Float32(host[base + 3]))
            end
        end
    end
end

function run_task!(results, task)
    threads = 256
    blocks = cld(IMAGE_SIZE * IMAGE_SIZE, threads)
    CUDA.synchronize()
    t0 = time_ns()
    if task == 1
        @cuda threads=threads blocks=blocks task1_kernel!(results)
    else
        @cuda threads=threads blocks=blocks task2_kernel!(results)
    end
    CUDA.synchronize()
    return (time_ns() - t0) * 1e-9
end

function main()
    results = CUDA.zeros(Float64, IMAGE_SIZE * IMAGE_SIZE * 3)

    println("task1: image size = $(IMAGE_SIZE)  x  $(IMAGE_SIZE)  pixels")
    dt1 = run_task!(results, 1)
    @printf("Total kernel execution time (task1) %f (s)\n", dt1)
    write_output("Output_task1.txt", "###output data:(alpha,  beta,  redshift)", Array(results))

    println("task2: image size = $(IMAGE_SIZE)  x  $(IMAGE_SIZE)  pixels")
    dt2 = run_task!(results, 2)
    @printf("Total kernel execution time (task2) %f (s)\n", dt2)
    write_output("Output_task2.txt", "###output data:(alpha,  beta, Luminosity (erg/sec))", Array(results))
    return 0
end

exit(main())
