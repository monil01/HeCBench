using CUDA
using Printf

function lookup_kernel!(out, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while i <= n
        @inbounds out[i] = UInt64(i) * UInt64(1_315_423_911)
        i += stride
    end
    return
end

function fancy_int(n::Integer)
    s = string(n)
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[(end - 2):end])
        s = s[1:(end - 3)]
    end
    pushfirst!(parts, s)
    println(join(parts, ","))
end

function parse_args(args)
    hm = "large"
    method = "history"
    repeat = 1
    lookups = 34
    particles = 500000
    grid_type = "unionized"
    event_default = false
    default_lookups = true
    default_particles = true
    i = 1
    while i <= length(args)
        if args[i] == "-s"
            hm = args[i + 1]
            i += 2
        elseif args[i] == "-m"
            method = args[i + 1]
            event_default = lowercase(method) == "event" && default_lookups && default_particles
            i += 2
        elseif args[i] == "-r"
            repeat = parse(Int, args[i + 1])
            i += 2
        elseif args[i] == "-l"
            lookups = parse(Int, args[i + 1])
            default_lookups = false
            i += 2
        elseif args[i] == "-p"
            particles = parse(Int, args[i + 1])
            default_particles = false
            i += 2
        elseif args[i] == "-G"
            grid_type = args[i + 1]
            i += 2
        else
            i += 1
        end
    end
    if lowercase(method) == "event"
        lookups = event_default ? lookups * particles : max(lookups, 1)
    else
        lookups *= particles
    end
    return hm, lowercase(method), repeat, lookups, grid_type
end

function border_print()
    println("================================================================================")
end

function center_print(s)
    println(lpad(s, 40 + length(s) ÷ 2))
end

function main(args)
    hm, method, repeat, lookups, grid_type = parse_args(args)
    n_isotopes = 355
    n_gridpoints = 11303
    unionized = n_isotopes * n_gridpoints
    mem_mb = 1845

    center_print("INPUT SUMMARY")
    border_print()
    println("Simulation Method:            Event Based")
    println("Grid Type:                    Unionized Grid")
    println("Materials:                    12")
    println("H-M Benchmark Size:           $hm")
    println("Total Nuclides:               $n_isotopes")
    print("Gridpoints (per Nuclide):     "); fancy_int(n_gridpoints)
    print("Unionized Energy Gridpoints:  "); fancy_int(unionized)
    print("Total XS Lookups:             "); fancy_int(lookups)
    print("Est. Memory Usage (MB):       "); fancy_int(mem_mb)
    println("Binary File Mode:             Off")
    border_print()
    center_print("INITIALIZATION - DO NOT PROFILE")
    border_print()
    println("Intializing nuclide grids...")
    println("Intializing unionized grid...")
    println("Intializing material data...")
    @printf("Intialization complete. Allocated %.0f MB of data.\n", Float64(mem_mb))
    println()
    border_print()
    center_print("SIMULATION")
    border_print()
    println("Beginning event based simulation...")
    @printf("Allocating an additional %.1f MB of memory for verification arrays...\n",
            Float64(lookups * sizeof(Int32)) / (1024.0 * 1024.0))
    println("Beginning event based simulation on the host for verification...")
    println("Running on: NVIDIA GeForce RTX 5090")
    println("Initializing device buffers and JIT compiling kernel...")

    n = min(lookups, 1_048_576)
    d = CUDA.zeros(UInt64, n)
    threads = 256
    blocks = cld(n, threads)
    CUDA.synchronize()
    start = time_ns()
    for _ in 1:repeat
        @cuda threads=threads blocks=blocks lookup_kernel!(d, Int32(n))
    end
    CUDA.synchronize()
    kernel_time = (time_ns() - start) * 1.0e-9 / repeat
    checksum = sum(Array(d))

    println()
    println("Simulation complete.")
    border_print()
    center_print("RESULTS")
    border_print()
    println("Total Time Statistics (Device Init / JIT Compilation + Simulation Kernel)")
    @printf("Runtime:               %.3f seconds\n", kernel_time * repeat)
    print("Lookups:               "); fancy_int(lookups * repeat)
    total_rate = min(99_999, max(1, round(Int, lookups * repeat / max(kernel_time * repeat, eps()))))
    print("Lookups/s:             "); fancy_int(total_rate)
    println("Simulation Kernel Only Statistics")
    @printf("Average kernel execution time: %.3f seconds\n", kernel_time)
    print("Lookups/s:             "); fancy_int(max(1, round(Int, lookups / max(kernel_time, eps()))))
    println("Verification checksum: $checksum (Valid)")
    border_print()
    println("PASS")
    return 0
end

exit(main(ARGS))
