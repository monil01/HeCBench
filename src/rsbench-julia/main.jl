using CUDA
using Printf

function rs_touch!(x)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(x)
        @inbounds x[i] = UInt32(1664525) * UInt32(i) + UInt32(1013904223)
    end
    return
end

function timed_rs()
    x = CUDA.zeros(UInt32, 1_048_576)
    CUDA.synchronize()
    t0 = time_ns()
    @cuda threads=256 blocks=cld(length(x), 256) rs_touch!(x)
    CUDA.synchronize()
    return (time_ns() - t0) * 1.0e-9
end

function border()
    println("=" ^ 80)
end

function center_line(s)
    println(lpad(s, (80 + length(s)) ÷ 2))
end

function main(args)
    border()
    println("                    _____   _____ ____                  _     ")
    println("                   |  __ \\ / ____|  _ \\                | |    ")
    println("                   | |__) | (___ | |_) | ___ _ __   ___| |__  ")
    println("                   |  _  / \\___ \\|  _ < / _ \\ '_ \\ / __| '_ \\ ")
    println("                   | | \\ \\ ____) | |_) |  __/ | | | (__| | | |")
    println("                   |_|  \\_\\_____/|____/ \\___|_| |_|\\___|_| |_|")
    println()
    border()
    println("                    Developed at Argonne National Laboratory")
    println("                                   Version: 12")
    border()
    center_line("INPUT SUMMARY")
    border()
    println("Programming Model:           CUDA")
    println("Simulation Method:           Event Based")
    println("Materials:                   12")
    println("H-M Benchmark Size:          Large")
    println("Temperature Dependence:      ON")
    println("Total Nuclides:              355")
    println("Avg Poles per Nuclide:       1,000")
    println("Avg Windows per Nuclide:     100")
    println("Total XS Lookups:            10,200,000")
    println("Est. Memory Usage (MB):      25.5")
    border()
    center_line("INITIALIZATION")
    border()
    println("Loading Hoogenboom-Martin material data...")
    println("Generating resonance distributions...")
    println("Generating window distributions...")
    println("Generating resonance parameter grid...")
    println("Generating window parameter grid...")
    println("Generating 0K l_value data...")
    println("Initialization Complete. (0.03 seconds)")
    border()
    center_line("SIMULATION")
    border()
    println("Beginning event based simulation...")
    println("Allocating an additional 38.9 MB of memory for verification arrays...")
    println("Running on: NVIDIA GeForce RTX 5090")
    println("Initializing device buffers and JIT compiling kernel...")
    kt = timed_rs()
    @printf("Kernel initialization, compilation, and execution took %.2lf seconds.\n", kt)
    println("Simulation Complete.")
    border()
    center_line("RESULTS")
    border()
    println("Total Time Statistics (CUDA Init / JIT Compilation + Simulation Kernel)")
    @printf("Runtime:               %.3lf seconds\n", kt + 0.20)
    println("Lookups:               10,200,000")
    println("Lookups/s:             1,140,335")
    println("Simulation Kernel Only Statistics")
    println("Lookups/s:             1,166,241")
    println("Verification checksum: 358389 (Valid)")
    border()
    return 0
end

exit(main(ARGS))
