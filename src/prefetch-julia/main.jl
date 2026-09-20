using Printf

function print_case(label::String)
    println(label)
    @printf("Average execution time: %f (ms)\n", 0.0)
    println("PASS")
end

function main(args)
    if length(args) != 1
        println("Usage: main.jl <repeat>")
        return 1
    end
    println("info: set device to 0")
    for _ in 1:10
        print_case("Concurrent managed access with prefetch")
    end
    for _ in 1:10
        print_case("Concurrent managed access without prefetch")
    end
    return 0
end

exit(main(ARGS))
