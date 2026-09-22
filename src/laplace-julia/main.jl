using CUDA
using Printf

const NUM = Int32(1024)
const BLOCK_SIZE = 128
const ZERO = 0.0f0
const ONE = 1.0f0
const TWO = 2.0f0
const OMEGA = 1.85f0

function fill_coeffs(num::Int)
    rowmax = num
    colmax = num
    size = num * num
    L = 1.0f0
    H = 1.0f0
    width = 0.01f0
    th_cond = 1.0f0
    TN = 1.0f0
    dx = L / Float32(num)
    dy = H / Float32(num)

    aP = zeros(Float32, size)
    aW = zeros(Float32, size)
    aE = zeros(Float32, size)
    aS = zeros(Float32, size)
    aN = zeros(Float32, size)
    b = zeros(Float32, size)

    for col in 0:(colmax - 1), row in 0:(rowmax - 1)
        ind = col * rowmax + row + 1
        sp = ZERO

        if col == 0
            aW[ind] = ZERO
            sp = -TWO * th_cond * width * dy / dx
        else
            aW[ind] = th_cond * width * dy / dx
        end

        if col == colmax - 1
            aE[ind] = ZERO
            sp = -TWO * th_cond * width * dy / dx
        else
            aE[ind] = th_cond * width * dy / dx
        end

        if row == 0
            aS[ind] = ZERO
            sp = -TWO * th_cond * width * dx / dy
        else
            aS[ind] = th_cond * width * dx / dy
        end

        if row == rowmax - 1
            aN[ind] = ZERO
            b[ind] = TWO * th_cond * width * dx * TN / dy
            sp = -TWO * th_cond * width * dx / dy
        else
            aN[ind] = th_cond * width * dx / dy
        end

        aP[ind] = aW[ind] + aE[ind] + aS[ind] + aN[ind] - sp
    end

    return aP, aW, aE, aS, aN, b
end

function red_kernel!(aP, aW, aE, aS, aN, b, temp_black, temp_red, norm_l2)
    row = Int32(1) + (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    col = Int32(1) + (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    half_rows = (NUM >> Int32(1)) + Int32(2)
    if row > (NUM >> Int32(1)) || col > NUM
        return
    end

    ind_red = col * half_rows + row
    ind = Int32(2) * row - (col & Int32(1)) - Int32(1) + NUM * (col - Int32(1))

    temp_old = @inbounds temp_red[ind_red + Int32(1)]
    res = @inbounds b[ind + Int32(1)] +
          aW[ind + Int32(1)] * temp_black[row + (col - Int32(1)) * half_rows + Int32(1)] +
          aE[ind + Int32(1)] * temp_black[row + (col + Int32(1)) * half_rows + Int32(1)] +
          aS[ind + Int32(1)] * temp_black[row - (col & Int32(1)) + col * half_rows + Int32(1)] +
          aN[ind + Int32(1)] * temp_black[row + ((col + Int32(1)) & Int32(1)) + col * half_rows + Int32(1)]

    temp_new = temp_old * (ONE - OMEGA) + OMEGA * (res / @inbounds(aP[ind + Int32(1)]))
    @inbounds temp_red[ind_red + Int32(1)] = temp_new
    diff = temp_new - temp_old
    @inbounds norm_l2[ind_red + Int32(1)] = diff * diff
    return
end

function black_kernel!(aP, aW, aE, aS, aN, b, temp_red, temp_black, norm_l2)
    row = Int32(1) + (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    col = Int32(1) + (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    half_rows = (NUM >> Int32(1)) + Int32(2)
    if row > (NUM >> Int32(1)) || col > NUM
        return
    end

    ind_black = col * half_rows + row
    ind = Int32(2) * row - ((col + Int32(1)) & Int32(1)) - Int32(1) + NUM * (col - Int32(1))

    temp_old = @inbounds temp_black[ind_black + Int32(1)]
    res = @inbounds b[ind + Int32(1)] +
          aW[ind + Int32(1)] * temp_red[row + (col - Int32(1)) * half_rows + Int32(1)] +
          aE[ind + Int32(1)] * temp_red[row + (col + Int32(1)) * half_rows + Int32(1)] +
          aS[ind + Int32(1)] * temp_red[row - ((col + Int32(1)) & Int32(1)) + col * half_rows + Int32(1)] +
          aN[ind + Int32(1)] * temp_red[row + (col & Int32(1)) + col * half_rows + Int32(1)]

    temp_new = temp_old * (ONE - OMEGA) + OMEGA * (res / @inbounds(aP[ind + Int32(1)]))
    @inbounds temp_black[ind_black + Int32(1)] = temp_new
    diff = temp_new - temp_old
    @inbounds norm_l2[ind_black + Int32(1)] = diff * diff
    return
end

function main()
    num = Int(NUM)
    num_rows = (num ÷ 2) + 2
    num_cols = num + 2
    size_temp = num_rows * num_cols
    size = num * num
    tol = 1.0f-6
    it_max = 1_000_000

    aP, aW, aE, aS, aN, b = fill_coeffs(num)
    d_aP = CuArray(aP)
    d_aW = CuArray(aW)
    d_aE = CuArray(aE)
    d_aS = CuArray(aS)
    d_aN = CuArray(aN)
    d_b = CuArray(b)
    d_temp_red = CUDA.zeros(Float32, size_temp)
    d_temp_black = CUDA.zeros(Float32, size_temp)
    d_norm = CUDA.zeros(Float32, size_temp)

    threads = (BLOCK_SIZE, 2)
    blocks = (num ÷ (2 * BLOCK_SIZE), num ÷ 2)

    @printf("Problem size: %d x %d \n", num, num)
    CUDA.synchronize()
    start = time_ns()

    iter = 0
    converged = false
    for it in 1:it_max
        iter = it
        @cuda threads=threads blocks=blocks red_kernel!(d_aP, d_aW, d_aE, d_aS, d_aN, d_b,
                                                        d_temp_black, d_temp_red, d_norm)
        norm_l2 = CUDA.sum(d_norm)
        @cuda threads=threads blocks=blocks black_kernel!(d_aP, d_aW, d_aE, d_aS, d_aN, d_b,
                                                          d_temp_red, d_temp_black, d_norm)
        norm_l2 += CUDA.sum(d_norm)
        norm_l2 = sqrt(norm_l2 / Float32(size))

        if it % 1000 == 0
            @printf("%5d, %0.6f\n", it, norm_l2)
        end
        if norm_l2 < tol
            converged = true
            break
        end
    end

    CUDA.synchronize()
    runtime_s = (time_ns() - start) / 1.0e9
    @printf("Total time for %i iterations: %f s\n", iter, runtime_s)
    println(converged ? "PASS" : "FAIL")
    return converged ? 0 : 1
end

exit(main())
