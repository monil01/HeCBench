using CUDA
using Printf

const NUM_THREADS = 128
const NUM_BLOCKS = 256
const RAN = Float32[
    0.656619,0.891183,0.488144,0.992646,0.373326,0.531378,0.181316,0.501944,0.422195,
    0.660427,0.673653,0.95733,0.191866,0.111216,0.565054,0.969166,0.0237439,0.870216,
    0.0268766,0.519529,0.192291,0.715689,0.250673,0.933865,0.137189,0.521622,0.895202,
    0.942387,0.335083,0.437364,0.471156,0.14931,0.135864,0.532498,0.725789,0.398703,
    0.358419,0.285279,0.868635,0.626413,0.241172,0.978082,0.640501,0.229849,0.681335,
    0.665823,0.134718,0.0224933,0.262199,0.116515,0.0693182,0.85293,0.180331,0.0324186,
    0.733926,0.536517,0.27603,0.368458,0.0128863,0.889206,0.866021,0.254247,0.569481,
    0.159265,0.594364,0.3311,0.658613,0.863634,0.567623,0.980481,0.791832,0.152594,
    0.833027,0.191863,0.638987,0.669,0.772088,0.379818,0.441585,0.48306,0.608106,
    0.175996,0.00202556,0.790224,0.513609,0.213229,0.10345,0.157337,0.407515,0.407757,
    0.0526927,0.941815,0.149972,0.384374,0.311059,0.168534,0.896648]

function c_srand(seed::UInt32)
    ccall(:srand, Cvoid, (Cuint,), seed)
end

function c_rand()
    ccall(:rand, Cint, ())
end

function interp_host(d::NTuple{3,Int32}, f::Vector{UInt8}, x::Float32, y::Float32, z::Float32)
    ix = Int32(floor(x)); dx1 = x - ix; dx2 = 1f0 - dx1
    iy = Int32(floor(y)); dy1 = y - iy; dy2 = 1f0 - dy1
    iz = Int32(floor(z)); dz1 = z - iz; dz2 = 1f0 - dz1
    base = Int(ix + d[1] * (iy - 1 + d[2] * (iz - 1)))
    k222 = Float32(f[base]); k122 = Float32(f[base + 1])
    k212 = Float32(f[base + Int(d[1])]); k112 = Float32(f[base + Int(d[1]) + 1])
    base += Int(d[1] * d[2])
    k221 = Float32(f[base]); k121 = Float32(f[base + 1])
    k211 = Float32(f[base + Int(d[1])]); k111 = Float32(f[base + Int(d[1]) + 1])
    return (((k222*dx2+k122*dx1)*dy2 + (k212*dx2+k112*dx1)*dy1))*dz2 +
           (((k221*dx2+k121*dx1)*dy2 + (k211*dx2+k111*dx1)*dy1))*dz1
end

function interp_device(d1::Int32, d2::Int32, f, x::Float32, y::Float32, z::Float32)
    ix = Int32(floor(x)); dx1 = x - ix; dx2 = 1f0 - dx1
    iy = Int32(floor(y)); dy1 = y - iy; dy2 = 1f0 - dy1
    iz = Int32(floor(z)); dz1 = z - iz; dz2 = 1f0 - dz1
    base = ix + d1 * (iy - Int32(1) + d2 * (iz - Int32(1)))
    k222 = Float32(f[base]); k122 = Float32(f[base + Int32(1)])
    k212 = Float32(f[base + d1]); k112 = Float32(f[base + d1 + Int32(1)])
    base += d1 * d2
    k221 = Float32(f[base]); k121 = Float32(f[base + Int32(1)])
    k211 = Float32(f[base + d1]); k111 = Float32(f[base + d1 + Int32(1)])
    return (((k222*dx2+k122*dx1)*dy2 + (k212*dx2+k112*dx1)*dy1))*dz2 +
           (((k221*dx2+k121*dx1)*dy2 + (k211*dx2+k111*dx1)*dy1))*dz1
end

function spm_kernel!(M, data_size::Int32, g, f, dg1::Int32, dg2::Int32, dg3::Int32,
                     df1::Int32, df2::Int32, df3::Int32, ivf, ivg, thresh)
    idx0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    stride = blockDim().x * gridDim().x
    x_datasize = dg1 - Int32(2)
    y_datasize = dg2 - Int32(2)
    i = idx0
    while i < data_size
        ii = Float32(i)
        xx = Float32(i % x_datasize) + 1f0
        yy = Float32((i ÷ x_datasize) % y_datasize) + 1f0
        zz = floor(ii / Float32(x_datasize)) / Float32(y_datasize) + 1f0
        r = RAN[(i % Int32(97)) + Int32(1)]
        rx = xx + r; ry = yy + r; rz = zz + r
        xp = M[1]*rx + M[5]*ry + M[9]*rz + M[13]
        yp = M[2]*rx + M[6]*ry + M[10]*rz + M[14]
        zp = M[3]*rx + M[7]*ry + M[11]*rz + M[15]
        out = i + Int32(1)
        if zp >= 1f0 && zp < Float32(df3) && yp >= 1f0 && yp < Float32(df2) && xp >= 1f0 && xp < Float32(df1)
            ivf[out] = UInt8(floor(interp_device(df1, df2, f, xp, yp, zp) + 0.5f0))
            ivg[out] = UInt8(floor(interp_device(dg1, dg2, g, rx, ry, rz) + 0.5f0))
            thresh[out] = true
        else
            ivf[out] = UInt8(0)
            ivg[out] = UInt8(0)
            thresh[out] = false
        end
        i += stride
    end
    return
end

function spm_reference(M, data_size, g, f, dg, df)
    ivf = Vector{UInt8}(undef, data_size)
    ivg = Vector{UInt8}(undef, data_size)
    thresh = Vector{Bool}(undef, data_size)
    x_datasize = dg[1] - 2
    y_datasize = dg[2] - 2
    for i0 in 0:data_size-1
        xx = Float32(i0 % x_datasize) + 1f0
        yy = Float32((i0 ÷ x_datasize) % y_datasize) + 1f0
        zz = floor(Float32(i0) / Float32(x_datasize)) / Float32(y_datasize) + 1f0
        r = RAN[(i0 % 97) + 1]
        rx = xx + r; ry = yy + r; rz = zz + r
        xp = M[1]*rx + M[5]*ry + M[9]*rz + M[13]
        yp = M[2]*rx + M[6]*ry + M[10]*rz + M[14]
        zp = M[3]*rx + M[7]*ry + M[11]*rz + M[15]
        i = i0 + 1
        if zp >= 1f0 && zp < Float32(df[3]) && yp >= 1f0 && yp < Float32(df[2]) && xp >= 1f0 && xp < Float32(df[1])
            ivf[i] = UInt8(floor(interp_host(df, f, xp, yp, zp) + 0.5f0))
            ivg[i] = UInt8(floor(interp_host(dg, g, rx, ry, rz) + 0.5f0))
            thresh[i] = true
        else
            ivf[i] = 0x00; ivg[i] = 0x00; thresh[i] = false
        end
    end
    return ivf, ivg, thresh
end

function main()
    length(ARGS) == 2 || begin
        println("Usage: main.jl <dimension> <repeat>")
        exit(1)
    end
    v = parse(Int32, ARGS[1])
    repeat = parse(Int, ARGS[2])
    dg = (v, v, v)
    df = (v, v, v)
    data_size = Int((v + 1) * (v + 1) * (v + 5))
    vol_size = Int(v * v * v)

    c_srand(UInt32(123))
    M = Vector{Float32}(undef, 16)
    rand_max = Float32(2147483647)
    for i in 1:16
        M[i] = Float32(c_rand()) / rand_max
    end
    g = Vector{UInt8}(undef, data_size)
    f = Vector{UInt8}(undef, data_size)
    for i in 1:data_size
        g[i] = UInt8(mod(c_rand(), 256))
        f[i] = UInt8(mod(c_rand(), 256))
    end

    d_M = CuArray(M)
    d_g = CuArray(g)
    d_f = CuArray(f)
    d_ivf = CUDA.zeros(UInt8, vol_size)
    d_ivg = CUDA.zeros(UInt8, vol_size)
    d_thresh = CUDA.zeros(Bool, vol_size)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:repeat
        @cuda threads=NUM_THREADS blocks=NUM_BLOCKS spm_kernel!(
            d_M, Int32(vol_size), d_g, d_f, dg[1], dg[2], dg[3], df[1], df[2], df[3],
            d_ivf, d_ivg, d_thresh)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - t0) * 1e-6 / repeat
    @printf("Average kernel execution time: %f (ms)\n", elapsed_ms)

    ivf_d = Array(d_ivf)
    ivg_d = Array(d_ivg)
    thresh_d = Array(d_thresh)
    hist_d = zeros(Int32, 65536)
    count = 0
    for i in 1:vol_size
        if thresh_d[i]
            hist_d[Int(ivf_d[i]) + Int(ivg_d[i]) * 256 + 1] += 1
            count += 1
        end
    end
    println("Device count: $count")

    ivf_h, ivg_h, thresh_h = spm_reference(M, vol_size, g, f, dg, df)
    hist_h = zeros(Int32, 65536)
    count = 0
    for i in 1:vol_size
        if thresh_h[i]
            hist_h[Int(ivf_h[i]) + Int(ivg_h[i]) * 256 + 1] += 1
            count += 1
        end
    end
    println("Host count: $count")
    max_diff = maximum(abs.(hist_h .- hist_d); init=0)
    println("Maximum difference $max_diff")
    println(max_diff == 0 ? "PASS" : "FAIL")
end

main()
