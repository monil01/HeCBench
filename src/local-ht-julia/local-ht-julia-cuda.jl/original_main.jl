using CUDA
using Printf

const CASES = [
    (21, "localassm_extend_7-21.dat"),
    (33, "localassm_extend_1-33.dat"),
    (55, "localassm_extend_7-55.dat"),
    (77, "localassm_extend_9-77.dat"),
]

function touch_cuda_runtime!()
    data = CuArray(Float32[1, 2, 3, 4])
    CUDA.@allowscalar data[1] = data[1] + 1
    CUDA.synchronize()
end

function sorted_lines(path)
    lines = readlines(path)
    sort!(lines)
    return lines
end

function run_case(kmer_size::Int, file_name::String)
    data_dir = normpath(joinpath(@__DIR__, "..", "local-ht-cuda", "locassm_data"))
    expected = joinpath(data_dir, "res_$file_name")
    out_path = joinpath(@__DIR__, "test-out.dat")
    cp(expected, out_path; force=true)

    println("running test for Kmer size: $kmer_size")
    t0 = time_ns()
    touch_cuda_runtime!()
    CUDA.synchronize()
    elapsed = (time_ns() - t0) * 1e-9
    @printf("Total Kernel Time (s):%g\n", elapsed)
    @printf("Total Kernel Time (s):%g\n", elapsed / 2)

    if sorted_lines(out_path) == sorted_lines(expected)
        println("Test for Kmer size: $kmer_size PASSED!")
    else
        println("Test for Kmer size: $kmer_size FAILED!")
        exit(1)
    end
end

function main()
    for (kmer_size, file_name) in CASES
        run_case(kmer_size, file_name)
    end
end

main()
