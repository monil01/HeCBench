using CUDA
using Printf

# Julia port of aidw-cuda: adaptive IDW interpolation.

const A1 = 1.5f0
const A2 = 2f0
const A3 = 2.5f0
const A4 = 3f0
const A5 = 3.5f0
const R_MIN = 0f0
const R_MAX = 2f0
const BLOCK_SIZE = 256
const EPS = 1f0

function aidw_kernel!(dx, dy, dz, dnum::Int32,
                     ix, iy, iz, inum::Int32,
                     area::Float32, avg_dist)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if tid > inum
        return
    end
    @inbounds r_obs = avg_dist[tid]
    r_exp = 0.5f0 / sqrt(Float32(dnum) / area)
    R_S0 = r_obs / r_exp

    u_R = 0f0
    if R_S0 >= R_MIN
        u_R = 0.5f0 - 0.5f0 * cos(3.1415926f0 / R_MAX * (R_S0 - R_MIN))
    end
    if R_S0 >= R_MAX
        u_R = 1f0
    end

    alpha = 1f0
    if u_R >= 0f0 && u_R <= 0.1f0;  alpha = A1;  end
    if u_R > 0.1f0 && u_R <= 0.3f0; alpha = A1 * (1f0 - 5f0*(u_R-0.1f0)) + A2 * 5f0*(u_R-0.1f0); end
    if u_R > 0.3f0 && u_R <= 0.5f0; alpha = A3 * 5f0*(u_R-0.3f0) + A1 * (1f0 - 5f0*(u_R-0.3f0)); end
    if u_R > 0.5f0 && u_R <= 0.7f0; alpha = A3 * (1f0 - 5f0*(u_R-0.5f0)) + A4 * 5f0*(u_R-0.5f0); end
    if u_R > 0.7f0 && u_R <= 0.9f0; alpha = A5 * 5f0*(u_R-0.7f0) + A4 * (1f0 - 5f0*(u_R-0.7f0)); end
    if u_R > 0.9f0 && u_R <= 1f0;   alpha = A5; end
    alpha *= 0.5f0

    sum = 0f0
    z = 0f0
    @inbounds ixt = ix[tid]
    @inbounds iyt = iy[tid]
    j = Int32(1)
    while j <= dnum
        @inbounds dxj = dx[j]
        @inbounds dyj = dy[j]
        @inbounds dzj = dz[j]
        ddx = ixt - dxj
        ddy = iyt - dyj
        dist = ddx*ddx + ddy*ddy
        t = 1f0 / (dist)^(alpha)
        sum += t
        z += dzj * t
        j += Int32(1)
    end
    @inbounds iz[tid] = z / sum
    return
end

function aidw_kernel_tiled!(dx, dy, dz, dnum::Int32,
                             ix, iy, iz, inum::Int32,
                             area::Float32, avg_dist)
    tid = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if tid > inum
        return
    end
    sdx = CuStaticSharedArray(Float32, BLOCK_SIZE)
    sdy = CuStaticSharedArray(Float32, BLOCK_SIZE)
    sdz = CuStaticSharedArray(Float32, BLOCK_SIZE)

    @inbounds r_obs = avg_dist[tid]
    r_exp = 0.5f0 / sqrt(Float32(dnum) / area)
    R_S0 = r_obs / r_exp
    u_R = 0f0
    if R_S0 >= R_MIN
        u_R = 0.5f0 - 0.5f0 * cos(3.1415926f0 / R_MAX * (R_S0 - R_MIN))
    end
    if R_S0 >= R_MAX; u_R = 1f0; end

    alpha = 0f0
    if u_R >= 0f0 && u_R <= 0.1f0;  alpha = A1;  end
    if u_R > 0.1f0 && u_R <= 0.3f0; alpha = A1 * (1f0 - 5f0*(u_R-0.1f0)) + A2 * 5f0*(u_R-0.1f0); end
    if u_R > 0.3f0 && u_R <= 0.5f0; alpha = A3 * 5f0*(u_R-0.3f0) + A1 * (1f0 - 5f0*(u_R-0.3f0)); end
    if u_R > 0.5f0 && u_R <= 0.7f0; alpha = A3 * (1f0 - 5f0*(u_R-0.5f0)) + A4 * 5f0*(u_R-0.5f0); end
    if u_R > 0.7f0 && u_R <= 0.9f0; alpha = A5 * 5f0*(u_R-0.7f0) + A4 * (1f0 - 5f0*(u_R-0.7f0)); end
    if u_R > 0.9f0 && u_R <= 1f0;   alpha = A5; end
    alpha *= 0.5f0

    @inbounds six_t = ix[tid]
    @inbounds siy_t = iy[tid]
    lid = threadIdx().x

    part = (dnum - Int32(1)) ÷ Int32(BLOCK_SIZE)
    sum_up = 0f0
    sum_dn = 0f0
    m = Int32(0)
    while m <= part
        num_threads = min(Int32(BLOCK_SIZE), dnum - Int32(BLOCK_SIZE) * m)
        if lid <= num_threads
            @inbounds sdx[lid] = dx[lid + Int32(BLOCK_SIZE) * m]
            @inbounds sdy[lid] = dy[lid + Int32(BLOCK_SIZE) * m]
            @inbounds sdz[lid] = dz[lid + Int32(BLOCK_SIZE) * m]
        end
        sync_threads()
        e = Int32(1)
        while e <= Int32(BLOCK_SIZE)
            @inbounds six_s = six_t - sdx[e]
            @inbounds siy_s = siy_t - sdy[e]
            dist = six_s*six_s + siy_s*siy_s
            t = 1f0 / (dist)^(alpha)
            sum_dn += t
            @inbounds sum_up += t * sdz[e]
            e += Int32(1)
        end
        sync_threads()
        m += Int32(1)
    end
    @inbounds iz[tid] = sum_up / sum_dn
    return
end

function reference_cpu(dx, dy, dz, dnum, ix, iy, inum, area, avg_dist)
    iz = zeros(Float32, inum)
    Threads.@threads for tid in 1:inum
        r_obs = avg_dist[tid]
        r_exp = 1f0 / (2f0 * sqrt(Float32(dnum) / area))
        R_S0 = r_obs / r_exp
        u_R = 0f0
        if R_S0 >= R_MIN
            u_R = 0.5f0 - 0.5f0 * cos(3.1415926f0 / R_MAX * (R_S0 - R_MIN))
        end
        if R_S0 >= R_MAX; u_R = 1f0; end
        alpha = 0f0
        if u_R >= 0f0 && u_R <= 0.1f0;  alpha = A1;  end
        if u_R > 0.1f0 && u_R <= 0.3f0; alpha = A1 * (1f0 - 5f0*(u_R-0.1f0)) + A2 * 5f0*(u_R-0.1f0); end
        if u_R > 0.3f0 && u_R <= 0.5f0; alpha = A3 * 5f0*(u_R-0.3f0) + A1 * (1f0 - 5f0*(u_R-0.3f0)); end
        if u_R > 0.5f0 && u_R <= 0.7f0; alpha = A3 * (1f0 - 5f0*(u_R-0.5f0)) + A4 * 5f0*(u_R-0.5f0); end
        if u_R > 0.7f0 && u_R <= 0.9f0; alpha = A5 * 5f0*(u_R-0.7f0) + A4 * (1f0 - 5f0*(u_R-0.7f0)); end
        if u_R > 0.9f0 && u_R <= 1f0;   alpha = A5; end
        alpha *= 0.5f0
        sum = 0f0
        z = 0f0
        for j in 1:dnum
            ddx = ix[tid] - dx[j]
            ddy = iy[tid] - dy[j]
            dist = ddx*ddx + ddy*ddy
            t = 1f0 / (dist^alpha)
            sum += t
            z += dz[j] * t
        end
        iz[tid] = z / sum
    end
    return iz
end

function main()
    if length(ARGS) != 3
        println("Usage: main.jl <pts> <check> <iterations>")
        return 1
    end
    numk = parse(Int, ARGS[1])
    check = parse(Int, ARGS[2])
    iterations = parse(Int, ARGS[3])

    dnum = numk * 1024
    inum = dnum
    width = 2000f0; height = 2000f0
    area = width * height

    state = UInt64(123)
    @inline function lcg_f32()
        state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
        return Float32((state >> 11) & UInt64(0xFFFFFF)) / Float32(1 << 24)
    end

    dx = Vector{Float32}(undef, dnum)
    dy = Vector{Float32}(undef, dnum)
    dz = Vector{Float32}(undef, dnum)
    ix = Vector{Float32}(undef, inum)
    iy = Vector{Float32}(undef, inum)
    avg_dist = Vector{Float32}(undef, dnum)

    for i in 1:dnum
        dx[i] = lcg_f32() * 1000f0
        dy[i] = lcg_f32() * 1000f0
        dz[i] = lcg_f32() * 1000f0
    end
    for i in 1:inum
        ix[i] = lcg_f32() * 1000f0
        iy[i] = lcg_f32() * 1000f0
    end
    for i in 1:dnum
        avg_dist[i] = lcg_f32() * 3f0
    end

    @printf("Size = : %d K \n", numk)
    @printf("dnum = : %d\ninum = : %d\n", dnum, inum)

    h_iz = zeros(Float32, inum)
    if check == 1
        println("Verification enabled")
        h_iz = reference_cpu(dx, dy, dz, dnum, ix, iy, inum, area, avg_dist)
    else
        println("Verification disabled")
    end

    d_dx = CuArray(dx); d_dy = CuArray(dy); d_dz = CuArray(dz)
    d_avg_dist = CuArray(avg_dist)
    d_ix = CuArray(ix); d_iy = CuArray(iy)
    d_iz = CUDA.zeros(Float32, inum)

    threads = BLOCK_SIZE
    blocks = cld(inum, BLOCK_SIZE)

    @cuda threads=threads blocks=blocks aidw_kernel!(d_dx, d_dy, d_dz,
        Int32(dnum), d_ix, d_iy, d_iz, Int32(inum), area, d_avg_dist)
    CUDA.synchronize()
    iz_gpu = Array(d_iz)
    if check == 1
        ok = all(abs.(iz_gpu .- h_iz) .<= EPS)
        println(ok ? "PASS" : "FAIL")
    end

    @cuda threads=threads blocks=blocks aidw_kernel_tiled!(d_dx, d_dy, d_dz,
        Int32(dnum), d_ix, d_iy, d_iz, Int32(inum), area, d_avg_dist)
    CUDA.synchronize()
    iz_gpu = Array(d_iz)
    if check == 1
        ok = all(abs.(iz_gpu .- h_iz) .<= EPS)
        println(ok ? "PASS" : "FAIL")
    end

    t0 = time_ns()
    for _ in 1:iterations
        @cuda threads=threads blocks=blocks aidw_kernel!(d_dx, d_dy, d_dz,
            Int32(dnum), d_ix, d_iy, d_iz, Int32(inum), area, d_avg_dist)
    end
    CUDA.synchronize()
    @printf("Average execution time of AIDW_Kernel       %f (s)\n",
            (time_ns() - t0) * 1e-9 / iterations)

    t0 = time_ns()
    for _ in 1:iterations
        @cuda threads=threads blocks=blocks aidw_kernel_tiled!(d_dx, d_dy, d_dz,
            Int32(dnum), d_ix, d_iy, d_iz, Int32(inum), area, d_avg_dist)
    end
    CUDA.synchronize()
    @printf("Average execution time of AIDW_Kernel_Tiled %f (s)\n",
            (time_ns() - t0) * 1e-9 / iterations)
    return 0
end

main()
