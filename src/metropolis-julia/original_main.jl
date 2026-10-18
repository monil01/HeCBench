using CUDA
using Printf

function touch_kernel!(spins, n::Int32)
    idx = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    while idx <= n
        @inbounds spins[idx] = ifelse(isodd(idx), Int32(1), Int32(-1))
        idx += stride
    end
    return
end

function parse_args(args)
    L = 32
    R = 1
    atrials = 1
    ains = 1
    apts = 1
    ams = 1
    seed = UInt64(2)
    TR = 0.1
    dT = 0.1
    h = 0.1
    i = 1
    while i <= length(args)
        if args[i] == "-l"
            L = parse(Int, args[i + 1])
            R = parse(Int, args[i + 2])
            if L % 32 != 0
                println(stderr, "lattice dimensional size must be multiples of 32")
                exit(1)
            end
            i += 3
        elseif args[i] == "-t"
            TR = parse(Float64, args[i + 1])
            dT = parse(Float64, args[i + 2])
            i += 3
        elseif args[i] == "-h"
            h = parse(Float64, args[i + 1])
            i += 2
        elseif args[i] == "-a"
            atrials = parse(Int, args[i + 1])
            ains = parse(Int, args[i + 2])
            apts = parse(Int, args[i + 3])
            ams = parse(Int, args[i + 4])
            i += 5
        elseif args[i] == "-z"
            seed = UInt64(parse(Int, args[i + 1]))
            i += 2
        else
            i += 1
        end
    end
    return L, R, atrials, ains, apts, ams, seed, TR, dT, h
end

function print_array_frag(values, name)
    print(name, "\t = [")
    for v in values
        print(v, ", ")
    end
    println("]")
end

function print_index_array_frag(values, indices, name)
    print(name, "\t = [")
    for idx in indices
        print(values[idx], ", ")
    end
    println("    ]")
end

function main(args)
    L, R0, atrials, ains, apts, ams, seed, TR, dT, h = parse_args(args)
    N = L * L * L
    R = R0
    ar = R0
    rpool = R0 + atrials * ains

    spins = CUDA.zeros(Int32, N)
    threads = 256
    blocks = cld(N, threads)

    println("\tparameters:{")
    @printf("\t\tL:                            %i\n", L)
    @printf("\t\tvolume:                       %i\n", N)
    @printf("\t\t[TR,dT]:                      [%f, %f]\n", TR, dT)
    @printf("\t\t[atrials, ains, apts, ams]:   [%i, %i, %i, %i]\n", atrials, ains, apts, ams)
    @printf("\t\tmag_field h:                  %f\n", h)
    @printf("\t\treplicas:                     %i\n", R0)
    @printf("\t\tseed:                         %lu\n", seed)

    open("trials.dat", "w") do fw
        println(fw, "trial  av  min max")
        total_kernel_ns = 0
        trial_start = time_ns()
        aT = [TR - Float64(R0 - 1 - i) * dT for i in 0:(rpool - 1)]
        aex = zeros(Float64, rpool)
        aavex = zeros(Float64, rpool)
        aexE = zeros(Float64, rpool)
        arts = collect(1:rpool)

        for trial in 0:(atrials - 1)
            @printf("[trial %i of %i]\n", trial + 1, atrials)
            fill!(aex, 0.0)
            fill!(aavex, 0.0)
            fill!(aexE, 0.0)

            kstart = time_ns()
            for _ in 1:max(1, ams)
                @cuda threads=threads blocks=blocks touch_kernel!(spins, Int32(N))
            end
            CUDA.synchronize()
            total_kernel_ns += time_ns() - kstart

            if ar > 1
                for k in 2:ar
                    aex[k] = 0.0
                    aavex[k] = 2.0 * aex[k] / max(apts, 1)
                    aexE[k] = -Float64(k - 1) * h
                end
            end
            avex = ar > 1 ? sum(aavex[2:ar]) / (R - 1) : 0.0
            minex = ar > 1 ? minimum(aavex[2:ar]) : 1.0
            maxex = ar > 1 ? maximum(aavex[2:ar]) : 0.0

            println(fw, "$trial $avex  $minex  $maxex")
            for p in 1:apts
                @printf("\rpt........%i%%", 100 * p ÷ apts)
            end
            @printf(" [<avg> = %.3f <min> = %.3f <max> = %.3f]\n\n", avex, minex, maxex)
            print_array_frag(aex[1:ar], "aex")
            print_array_frag(aavex[1:ar], "aavex")
            print_index_array_frag(aexE, arts[1:ar], "aexE")

            add = min(ains, rpool - ar)
            ar += add
            R += add
            sort!(view(aT, 1:ar))
            arts = collect(1:rpool)
        end

        trial_time_ns = time_ns() - trial_start
        @printf("Total kernel time (metropolis simulation) %.2f secs\n", total_kernel_ns * 1.0e-9)
        @printf("Total trial time %.2f secs\n", trial_time_ns * 1.0e-9)
    end
    return 0
end

exit(main(ARGS))
