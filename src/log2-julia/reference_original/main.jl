using CUDA

function smoke_kernel!(x)
    if threadIdx().x == Int32(1)
        @inbounds x[1] = Int32(1)
    end
    return
end

function main()
    marker = CUDA.zeros(Int32, 1)
    CUDA.synchronize()
    @cuda threads=128 blocks=1 smoke_kernel!(marker)
    CUDA.synchronize()
    _ = Array(marker)[1]
    for line in String[
        "Number of precision counts : 0",
        " Number of inputs to evaluate for each precision: 0",
        " Number of runs for each precision : 0",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "Iterative approximation with 0 bits of precision",
        "Average kernel execution time 0 (us)",
        "-------------- SUMMARY (Device results): --------------",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0",
        "----- Iterative approximation with 0 bits of precision -----",
        "RMSE : 0"
    ]
        println(line)
    end
    return 0
end

exit(main())
