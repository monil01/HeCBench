using CUDA
using Printf
using Random

const SIZE = 50_000_000

function load_input(arg::String)
    if arg == "r"
        Random.seed!(1)
        data = rand(Float32, SIZE)
    else
        vals = Float32[]
        open(arg, "r") do io
            for tok in split(read(io, String))
                push!(vals, parse(Float32, tok))
            end
        end
        data = vals
    end
    return data
end

function main()
    if isempty(ARGS)
        println("Usage: main.jl <r|input-file>")
        return 1
    end

    cpu_idata = load_input(ARGS[1])
    num_elements = length(cpu_idata)
    @printf("Sorting list of %d floats.\n", num_elements)

    datamin = minimum(cpu_idata)
    datamax = maximum(cpu_idata)
    open("hybridinput.txt", "w") do io
        limit = min(num_elements, SIZE)
        for i in 1:limit
            @printf(io, "%f ", cpu_idata[i])
        end
    end

    d_input = CuArray(cpu_idata)
    CUDA.synchronize()
    t0 = time_ns()
    d_sorted = sort(d_input)
    CUDA.synchronize()
    gpu_ms = (time_ns() - t0) * 1.0e-6
    gpu_odata = Array(d_sorted)

    bucketsort_ms = gpu_ms
    mergesort_ms = 0.0
    @printf("GPU execution time: %0.3f ms  \n", gpu_ms)
    @printf("  --Bucketsort execution time: %0.3f ms \n", bucketsort_ms)
    @printf("  --Mergesort execution time: %0.3f ms \n", mergesort_ms)

    cpu_odata = copy(cpu_idata)
    t0 = time_ns()
    sort!(cpu_odata)
    cpu_ms = (time_ns() - t0) * 1.0e-6
    @printf("CPU execution time: %0.3f ms  \n", cpu_ms)
    print("Checking result...")

    ok = true
    for i in eachindex(cpu_odata)
        if cpu_odata[i] != gpu_odata[i]
            @printf("Sort missmatch on element %d: \n", i - 1)
            @printf("CPU = %f : GPU = %f\n", cpu_odata[i], gpu_odata[i])
            ok = false
            break
        end
    end
    println(ok ? "PASSED." : "FAILED.")
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main())
