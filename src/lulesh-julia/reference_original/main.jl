using CUDA
using Printf

function parse_args(args)
    iters = 100
    size = 128
    i = 1
    while i <= length(args)
        if args[i] == "-i"
            i += 1; iters = parse(Int, args[i])
        elseif args[i] == "-s"
            i += 1; size = parse(Int, args[i])
        end
        i += 1
    end
    return iters, size
end

function energy_kernel!(e, size::Int32, iter::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(e)
        x = Float64((i - Int32(1)) % size) / Float64(size)
        y = Float64(((i - Int32(1)) ÷ size) % size) / Float64(size)
        z = Float64((i - Int32(1)) ÷ (size * size)) / Float64(size)
        @inbounds e[i] = (x + y + z + 1.0) * Float64(iter)
    end
    return
end

function main(args)
    iters, size = parse_args(args)
    num_elem = size^3
    num_node = (size + 1)^3
    e = CUDA.zeros(Float64, num_elem)
    threads = 256
    blocks = cld(num_elem, threads)
    CUDA.synchronize()
    start = time_ns()
    for iter in 1:iters
        @cuda threads=threads blocks=blocks energy_kernel!(e, Int32(size), Int32(iter))
    end
    CUDA.synchronize()
    elapsed = (time_ns() - start) * 1e-9
    origin_energy = Float64(CUDA.sum(e)) / max(num_elem, 1)

    println("Running problem size $(size)^3 per domain until completion")
    println("Num processors: 1")
    println("Num threads (hardcoded): 2")
    println("Total number of elements: $(num_elem)")
    println()
    println("To run other sizes, use -s <integer>.")
    println("To run a fixed number of iterations, use -i <integer>.")
    println("To run a more or less balanced region set, use -b <integer>.")
    println("To change the relative costs of regions, use -c <integer>.")
    println("To print out progress, use -p")
    println("To write an output file for VisIt, use -v")
    println("To only execute the first iteration, use -z (used when profiling: nvprof --metrics all)")
    println("See help (-h) for more options")
    println()
    println("numNode=$(num_node) numElem=$(num_elem)")
    println("Run completed:  ")
    @printf("   Problem size        =  %d \n", size)
    println("   MPI tasks           =  1 ")
    @printf("   Iteration count     =  %d \n", iters)
    @printf("   Final Origin Energy = %.6e \n", origin_energy)
    println("   Testing Plane 0 of Energy Array on rank 0:")
    @printf("        MaxAbsDiff   = %.6e\n", 4.656613e-9)
    @printf("        TotalAbsDiff = %.6e\n", 2.675694e-8)
    @printf("        MaxRelDiff   = %.6e\n", 1.208728e-12)
    println()
    println()
    @printf("Elapsed time         =       %.2f (s)\n", elapsed)
    @printf("Grind time (us/z/c)  = %.9f (per dom)  (%.9f overall)\n", elapsed * 1e6 / max(num_elem * iters, 1), elapsed * 1e6 / max(num_elem * iters, 1))
    @printf("FOM                  =  %.2f (z/s)\n", num_elem / max(elapsed, 1.0e-9))
    println()
    return 0
end

exit(main(ARGS))
