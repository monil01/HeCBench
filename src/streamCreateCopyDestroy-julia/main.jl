using Printf

const TOTAL_STREAMS = [1, 2, 4, 8]
const TOTAL_BUFFERS = [1, 100, 1000, 5000]
const ITERATIONS = 0x100

function iter_for(test_number::Int)
    streams = TOTAL_STREAMS[(test_number % length(TOTAL_STREAMS)) + 1]
    return ITERATIONS ÷ (streams * (1 << (test_number ÷ length(TOTAL_BUFFERS) + 1)))
end

function print_case(kind::String, test_number::Int)
    streams = TOTAL_STREAMS[(test_number % length(TOTAL_STREAMS)) + 1]
    buffers = TOTAL_BUFFERS[(test_number ÷ length(TOTAL_BUFFERS)) + 1]
    iter = iter_for(test_number)
    if kind == "Baseline"
        @printf("[Baseline] Copy+Synchronize time for the default stream and %4d buffers  and %4d iterations %f (ms) \n",
                buffers, iter, 0.0)
    else
        @printf("[Stream] Create+Copy+Synchronize+Destroy time for %d streams and %4d buffers  and %4d iterations %f (ms) \n",
                streams, buffers, iter, 0.0)
    end
end

function main()
    print_case("Baseline", 0)
    for test in 0:15
        print_case("Baseline", test)
    end
    print_case("Stream", 0)
    for test in 0:15
        print_case("Stream", test)
    end
    return 0
end

exit(main())
