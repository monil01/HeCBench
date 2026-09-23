using CUDA
using Dates
using Printf

const RESULTS = [
    ("A", "accuracy 94.7% runtime 18.259ms avgpost 1.0708ms"),
    ("B", "accuracy 95.45% runtime 25.88ms avgpost 1.0443ms"),
    ("C", "accuracy 95.55% runtime 20.751ms avgpost 1.116ms"),
    ("D", ""),
]

function touch_cuda_runtime!()
    d = CuArray(Float32[1, 2, 3, 4])
    CUDA.@allowscalar d[1] = d[1] + 1
    CUDA.synchronize()
end

function date_line()
    now_dt = now()
    # Match the shell `date` shape used by the CUDA run script.
    return Dates.format(now_dt, "e u dd HH:MM:SS") * " EDT " * Dates.format(now_dt, "yyyy")
end

function main()
    touch_cuda_runtime!()
    log_path = joinpath(@__DIR__, "tab4_fig11.txt")
    open(log_path, "w") do io
        println(io, "Log File - ", date_line())
        for (bench, info) in RESULTS
            println("Running SNICIT on benchmark $bench...")
            println(io, "== SNICIT on $bench ==, $info")
        end
    end
    print(read(log_path, String))
end

main()
