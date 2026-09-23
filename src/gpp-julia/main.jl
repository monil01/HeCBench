using CUDA
using Printf

const NSTART = 0
const NEND = 3

@inline function cmul(ar::Float64, ai::Float64, br::Float64, bi::Float64)
    return ar * br - ai * bi, ar * bi + ai * br
end

function solver_kernel!(
    number_bands::Int32, ngpown::Int32, ncouls::Int32,
    inv_igp_index, indinv, wx_array,
    wtilde_re, wtilde_im, aqsm_re, aqsm_im, aqsn_re, aqsn_im,
    ieps_re, ieps_im, vcoul, achtemp_re, achtemp_im)

    local_re0 = 0.0
    local_im0 = 0.0
    local_re1 = 0.0
    local_im1 = 0.0
    local_re2 = 0.0
    local_im2 = 0.0

    n1 = blockIdx().x - Int32(1)
    while n1 < number_bands
        my_igp = blockIdx().y - Int32(1)
        while my_igp < ngpown
            indigp = inv_igp_index[my_igp + Int32(1)]
            igp = indinv[indigp + Int32(1)]

            aqsm_idx = n1 * ncouls + igp + Int32(1)
            sch_r, sch_i = cmul(aqsm_re[aqsm_idx], -aqsm_im[aqsm_idx],
                                aqsn_re[aqsm_idx], aqsn_im[aqsm_idx])
            scale = 0.5 * vcoul[igp + Int32(1)]
            sch_r *= scale
            sch_i *= scale

            ig = threadIdx().x - Int32(1)
            while ig < ncouls
                eps_idx = my_igp * ncouls + ig + Int32(1)
                wt_r = wtilde_re[eps_idx]
                wt_i = wtilde_im[eps_idx]
                ie_r = ieps_re[eps_idx]
                ie_i = ieps_im[eps_idx]

                w = wx_array[1]
                wd_r = w - wt_r
                wd_i = -wt_i
                denom = wd_r * wd_r + wd_i * wd_i
                t_r, t_i = cmul(wt_r, wt_i, wd_r, -wd_i)
                del_r = t_r / denom
                del_i = t_i / denom
                arr_r, arr_i = cmul(del_r, del_i, ie_r, ie_i)
                arr_r, arr_i = cmul(arr_r, arr_i, sch_r, sch_i)
                local_re0 += arr_r
                local_im0 += arr_i

                w = wx_array[2]
                wd_r = w - wt_r
                wd_i = -wt_i
                denom = wd_r * wd_r + wd_i * wd_i
                t_r, t_i = cmul(wt_r, wt_i, wd_r, -wd_i)
                del_r = t_r / denom
                del_i = t_i / denom
                arr_r, arr_i = cmul(del_r, del_i, ie_r, ie_i)
                arr_r, arr_i = cmul(arr_r, arr_i, sch_r, sch_i)
                local_re1 += arr_r
                local_im1 += arr_i

                w = wx_array[3]
                wd_r = w - wt_r
                wd_i = -wt_i
                denom = wd_r * wd_r + wd_i * wd_i
                t_r, t_i = cmul(wt_r, wt_i, wd_r, -wd_i)
                del_r = t_r / denom
                del_i = t_i / denom
                arr_r, arr_i = cmul(del_r, del_i, ie_r, ie_i)
                arr_r, arr_i = cmul(arr_r, arr_i, sch_r, sch_i)
                local_re2 += arr_r
                local_im2 += arr_i

                ig += blockDim().x
            end
            my_igp += gridDim().y
        end
        n1 += gridDim().x
    end

    CUDA.@atomic achtemp_re[1] += local_re0
    CUDA.@atomic achtemp_im[1] += local_im0
    CUDA.@atomic achtemp_re[2] += local_re1
    CUDA.@atomic achtemp_im[2] += local_im1
    CUDA.@atomic achtemp_re[3] += local_re2
    CUDA.@atomic achtemp_im[3] += local_im2
    return
end

function parse_problem()
    if length(ARGS) == 0
        return 512, 2, 512, 20, "test"
    elseif length(ARGS) == 1
        if ARGS[1] == "benchmark"
            return 512, 2, 32768, 20, "benchmark"
        elseif ARGS[1] == "test"
            return 512, 2, 512, 20, "test"
        else
            println("Usage: ./main <test or benchmark>")
            println("Problem unrecognized, use 'test' or 'benchmark'")
            exit(0)
        end
    elseif length(ARGS) == 4
        return parse(Int, ARGS[1]), parse(Int, ARGS[2]), parse(Int, ARGS[3]), parse(Int, ARGS[4]), "test"
    else
        println("The correct form of input is : ")
        println(" ./main <number_bands> <number_valence_bands> <number_plane_waves> <nodes_per_mpi_group> ")
        exit(0)
    end
end

function correctness(problem::String, re::Float64, im::Float64)
    if problem == "benchmark"
        re_diff = re - -24852.551547
        im_diff = im - 2957453.638101
        if re_diff < 0.00001 && im_diff < 0.00001
            println("\nBenchmark result: SUCCESS")
        else
            println("\nBenchmark result: FAILURE")
        end
    else
        re_diff = re - -0.096066
        im_diff = im - 11.431852
        if re_diff < 0.00001 && im_diff < 0.00001
            println("\nTest result: SUCCESS")
        else
            println("\nTest result: FAILURE")
        end
    end
end

function main()
    number_bands, nvband, ncouls, nodes_per_group, problem = parse_problem()
    ngpown = ncouls ÷ nodes_per_group
    e_lk = 10.0
    dw = 1.0
    to1 = 1e-6
    e_n1kq = 6.0

    @printf("Sizeof(CustomComplex<dataType> = %d bytes\n", 16)
    @printf("number_bands = %d\t nvband = %d\t ncouls = %d\t nodes_per_group  = %d\t ngpown = %d\t nend = %d\t nstart = %d\n",
            number_bands, nvband, ncouls, nodes_per_group, ngpown, NEND, NSTART)

    start_total = time_ns()
    expr_re = 0.025
    expr_im = 0.025
    achtemp_re = zeros(Float64, NEND - NSTART)
    achtemp_im = zeros(Float64, NEND - NSTART)
    aqsm_re = fill(expr_re, number_bands * ncouls)
    aqsm_im = fill(expr_im, number_bands * ncouls)
    aqsn_re = fill(expr_re, number_bands * ncouls)
    aqsn_im = fill(expr_im, number_bands * ncouls)
    ieps_re = fill(expr_re, ngpown * ncouls)
    ieps_im = fill(expr_im, ngpown * ncouls)
    wtilde_re = fill(expr_re, ngpown * ncouls)
    wtilde_im = fill(expr_im, ngpown * ncouls)
    vcoul = [Float64(i - 1) * 0.025 for i in 1:ncouls]
    inv_igp_index = [Int32(ig * ncouls ÷ ngpown) for ig in 1:ngpown]
    indinv = [Int32(i - 1) for i in 1:(ncouls + 1)]
    indinv[end] = Int32(ncouls - 1)
    wx_array = zeros(Float64, NEND - NSTART)
    for iw in 0:(NEND - 1)
        wx_array[iw + 1] = e_lk - e_n1kq + dw * ((iw + 1) - 2)
        if wx_array[iw + 1] < to1
            wx_array[iw + 1] = to1
        end
    end

    mem_footprint = 2 * length(aqsm_re) * sizeof(Float64) +
                    4 * length(ieps_re) * sizeof(Float64) +
                    length(vcoul) * sizeof(Float64) +
                    3 * length(wx_array) * sizeof(Float64)
    @printf("Memory Foot Print = %.6f GBs\n", mem_footprint / 1024.0^3)

    d_aqsm_re = CuArray(aqsm_re)
    d_aqsm_im = CuArray(aqsm_im)
    d_aqsn_re = CuArray(aqsn_re)
    d_aqsn_im = CuArray(aqsn_im)
    d_ieps_re = CuArray(ieps_re)
    d_ieps_im = CuArray(ieps_im)
    d_wtilde_re = CuArray(wtilde_re)
    d_wtilde_im = CuArray(wtilde_im)
    d_vcoul = CuArray(vcoul)
    d_wx = CuArray(wx_array)
    d_inv = CuArray(inv_igp_index)
    d_indinv = CuArray(indinv)
    d_ach_re = CUDA.zeros(Float64, NEND - NSTART)
    d_ach_im = CUDA.zeros(Float64, NEND - NSTART)

    @printf("Launching a kernel with grid = (%d,%d,%d), and threads = (%d,%d,%d) \n",
            number_bands, ngpown, 1, 32, 1, 1)

    total_ktime = 0
    for _ in 1:10
        copyto!(d_ach_re, achtemp_re)
        copyto!(d_ach_im, achtemp_im)
        CUDA.synchronize()
        kstart = time_ns()
        @cuda blocks=(number_bands, ngpown, 1) threads=(32, 1, 1) solver_kernel!(
            Int32(number_bands), Int32(ngpown), Int32(ncouls),
            d_inv, d_indinv, d_wx,
            d_wtilde_re, d_wtilde_im, d_aqsm_re, d_aqsm_im, d_aqsn_re, d_aqsn_im,
            d_ieps_re, d_ieps_im, d_vcoul, d_ach_re, d_ach_im)
        CUDA.synchronize()
        total_ktime += time_ns() - kstart
    end

    @printf("Average kernel execution time %f (s)\n", (total_ktime * 1e-9) / 10)
    achtemp_re = Array(d_ach_re)
    achtemp_im = Array(d_ach_im)
    correctness(problem, achtemp_re[1], achtemp_im[1])

    println("\n Final achtemp")
    @printf("( %f, %f) \n", achtemp_re[1], achtemp_im[1])
    @printf("********** Total Time Taken **********= %g secs\n", (time_ns() - start_total) * 1e-9)
    return 0
end

exit(main())
