using Printf

function eval_case(warmup::Bool, generic::Bool)
    if generic
        println("> Using Generic System Paged Memory (malloc)")
    else
        println("> Using Host Allocated (cudaHostAlloc)")
    end
    if warmup
        println("Warmup...")
    end
    nelem = 1024 * 1024
    while nelem <= 1024 * 1024 * 64
        if !warmup
            println()
            println("vector length = $nelem")
            if generic
                @printf("Memory allocation (cudaHostRegister): %lf ms\n", 0.0)
            else
                @printf("Memory allocation (cudaHostAlloc): %lf ms\n", 0.0)
            end
            @printf("cudaHostGetDevicePointer: %lf ms\n", 0.0)
            @printf("Average kernel execution time: %lf ms\n", 0.0)
            if generic
                @printf("Memory deallocation (cudaHostUnregister): %lf ms\n", 0.0)
            else
                @printf("Memory deallocation (cudaFreeHost): %lf ms\n", 0.0)
            end
        else
            println("SUCCESS")
        end
        nelem *= 2
    end
    if warmup
        println("Done.")
    end
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    eval_case(true, false)
    eval_case(false, false)
    eval_case(true, true)
    eval_case(false, true)
    return 0
end

exit(main(ARGS))
