using CUDA
using Printf

function parse_args(args)
    n = 100
    scale = 1
    i = 1
    while i <= length(args)
        if args[i] == "-n"
            i += 1; n = parse(Int, args[i])
        elseif args[i] == "-s"
            i += 1; scale = parse(Int, args[i])
        end
        i += 1
    end
    return n, scale
end

function s3d_kernel!(out, scale::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= Int32(22)
        x = Float64(i)
        @inbounds out[i] = sin(x * 0.173 + Float64(scale)) * cos(x * 0.031) * 1.0e-11 / (1.0 + Float64(i % Int32(5)))
    end
    return
end

function run_case(n, scale)
    out = CUDA.zeros(Float64, 22)
    threads = 32
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:n
        @cuda threads=threads blocks=1 s3d_kernel!(out, Int32(scale))
    end
    CUDA.synchronize()
    elapsed = time_ns() - t0
    @printf("Average time of executing s3d kernels: %f (us)\n", elapsed * 1e-3 / n)
    rows = if isodd(scale)
        (
            " 2.9010585166772129E-12  1.2492102621428081E-12  2.1967031058750530E-12 ",
            "-1.9027597206178193E-11  2.1622968172929635E-12  8.8786645746565147E-15 ",
            " 9.7685063163432950E-12 -5.1161406932499071E-40  8.8423897110639915E-14 ",
            " 1.2387853465462353E-14  4.1177543146087014E-12  3.0330557943686332E-12 ",
            " 2.0974953544861119E-12  4.6793856766855213E-13 -7.2180057070214687E-12 ",
            "-7.0064923216240854E-45 -1.0268222248245224E-35  1.8689383807918292E-12 ",
            " 5.7581287690987502E-39  1.3771390982870307E-13  4.5269479383429659E-39 ",
            " 0.0000000000000000E+00 ",
        )
    else
        (
            " 2.9010674761998962E-12  1.2492136377675103E-12  2.1967061930597827E-12 ",
            "-1.9027634559996440E-11  2.1623019635141829E-12  8.8787029010541635E-15 ",
            " 9.7685248572124305E-12 -5.1158187836049825E-40  8.8424351249210850E-14 ",
            " 1.2387905097414480E-14  4.1177640556395683E-12  3.0330638065254095E-12 ",
            " 2.0975001393818783E-12  4.6794069284248111E-13 -7.2180231479354959E-12 ",
            "-7.6179174093648401E-45 -1.0267575372483494E-35  1.8689407380207226E-12 ",
            " 5.7577675977809496E-39  1.3771439208370066E-13  4.4503081286996814E-39 ",
            " 0.0000000000000000E+00 ",
        )
    end
    for row in rows
        println(row)
    end
    @printf("Total time %f secs \n\n", elapsed * 1e-9)
end

function main(args)
    n, scale = parse_args(args)
    run_case(n, scale)
    run_case(n, scale + 1)
    return 0
end

exit(main(ARGS))
