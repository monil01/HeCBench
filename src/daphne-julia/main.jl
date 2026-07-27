using CUDA
using Printf

# daphne Julia port: LiDAR-to-image projection surrogate.
# Mirrors Triton reference at ../daphne-triton/main.py — synthetic LCG input,
# per-point extrinsic rotation, radial+tangential undistortion, intrinsic
# projection, pt2>2.5 visibility filter.

const R00 = Float32(-0.9111348390579224)
const R01 = Float32( 0.0751304179430008)
const R02 = Float32(-0.4052018225193024)
const R10 = Float32(-0.3360927104949951)
const R11 = Float32(-0.7044632434844971)
const R12 = Float32( 0.6251187324523926)
const R20 = Float32(-0.2384843230247498)
const R21 = Float32( 0.7057529091835022)
const R22 = Float32( 0.6671117544174194)
const T0 = 0.1f0; const T1 = -0.2f0; const T2 = 0.3f0
const D0 = 0.03f0; const D1 = -0.15f0; const D2 = 0.001f0; const D3 = 0.001f0; const D4 = 0.05f0
const FX = 1200.0f0; const CX = 400.0f0; const FY = 1200.0f0; const CY = 300.0f0
const POINT_STEP = 8

function project_kernel!(cp, ox, oy, oz, ov, n::Int32)
    i = Int32((blockIdx().x - 1)) * Int32(blockDim().x) + Int32(threadIdx().x)
    if i > n
        return
    end
    base = (i - Int32(1)) * Int32(POINT_STEP) + Int32(1)
    @inbounds p0 = cp[base]
    @inbounds p1 = cp[base + 1]
    @inbounds p2 = cp[base + 2]
    pt0 = T0 + p0 * R00 + p1 * R01 + p2 * R02
    pt1 = T1 + p0 * R10 + p1 * R11 + p2 * R12
    pt2 = T2 + p0 * R20 + p1 * R21 + p2 * R22
    close = pt2 > 2.5f0
    denom = close ? pt2 : 1.0f0
    tmpx = pt0 / denom
    tmpy = pt1 / denom
    r2 = tmpx * tmpx + tmpy * tmpy
    tmpdist = 1.0f0 + D0 * r2 + D1 * r2 * r2 + D4 * r2 * r2 * r2
    ix = tmpx * tmpdist + 2.0f0 * D2 * tmpx * tmpy + D3 * (r2 + 2.0f0 * tmpx * tmpx)
    iy = tmpy * tmpdist + D2 * (r2 + 2.0f0 * tmpy * tmpy) + 2.0f0 * D3 * tmpx * tmpy
    ux = FX * ix + CX
    uy = FY * iy + CY
    xpix = ux + 0.5f0
    ypix = uy + 0.5f0
    @inbounds if close
        ox[i] = xpix; oy[i] = ypix; oz[i] = pt2 * 100.0f0; ov[i] = Int32(1)
    else
        ox[i] = 0.0f0; oy[i] = 0.0f0; oz[i] = 0.0f0; ov[i] = Int32(0)
    end
    return
end

function ref_project(p0::Float32, p1::Float32, p2::Float32)
    pt0 = T0 + p0 * R00 + p1 * R01 + p2 * R02
    pt1 = T1 + p0 * R10 + p1 * R11 + p2 * R12
    pt2 = T2 + p0 * R20 + p1 * R21 + p2 * R22
    close = pt2 > 2.5f0
    denom = close ? pt2 : 1.0f0
    tmpx = pt0 / denom
    tmpy = pt1 / denom
    r2 = tmpx * tmpx + tmpy * tmpy
    tmpdist = 1.0f0 + D0 * r2 + D1 * r2 * r2 + D4 * r2 * r2 * r2
    ix = tmpx * tmpdist + 2.0f0 * D2 * tmpx * tmpy + D3 * (r2 + 2.0f0 * tmpx * tmpx)
    iy = tmpy * tmpdist + D2 * (r2 + 2.0f0 * tmpy * tmpy) + 2.0f0 * D3 * tmpx * tmpy
    xpix = FX * ix + CX + 0.5f0
    ypix = FY * iy + CY + 0.5f0
    if close
        return xpix, ypix, pt2 * 100.0f0, Int32(1)
    else
        return 0.0f0, 0.0f0, 0.0f0, Int32(0)
    end
end

function main()
    n_batches = 1
    for k in 1:length(ARGS)-1
        if ARGS[k] == "-p"
            n_batches = parse(Int, ARGS[k+1])
        end
    end
    n_points = 100_000
    println("[note] synthetic $n_points points x $n_batches batches")

    cp = zeros(Float32, n_points * POINT_STEP)
    ref_x = zeros(Float32, n_points)
    ref_y = zeros(Float32, n_points)
    ref_z = zeros(Float32, n_points)
    ref_v = zeros(Int32,   n_points)

    ok_all = true
    total_us = 0.0

    for b in 0:n_batches-1
        s = UInt64(20260721 + (b + 1))
        @inbounds for i in 1:n_points
            u = ntuple(_ -> begin
                s = s * UInt64(6364136223846793005) + UInt64(1442695040888963407)
                Float32(((s >> 33) & UInt64(0x7fffffff))) / Float32(0x7fffffff)
            end, 4)
            base = (i - 1) * POINT_STEP + 1
            cp[base]     = (u[1] - 0.5f0) * 2.0f0
            cp[base + 1] = (u[2] - 0.5f0) * 2.0f0
            cp[base + 2] = u[3] * 10.0f0 + 15.0f0
            cp[base + 4] = u[4]
        end

        d_cp = CuArray(cp)
        d_x  = CUDA.zeros(Float32, n_points)
        d_y  = CUDA.zeros(Float32, n_points)
        d_z  = CUDA.zeros(Float32, n_points)
        d_v  = CUDA.zeros(Int32,   n_points)

        block = 256
        grid  = cld(n_points, block)

        CUDA.synchronize()
        t0 = time_ns()
        @cuda threads=block blocks=grid project_kernel!(d_cp, d_x, d_y, d_z, d_v, Int32(n_points))
        CUDA.synchronize()
        total_us += (time_ns() - t0) / 1e3

        h_x = Array(d_x); h_y = Array(d_y); h_z = Array(d_z); h_v = Array(d_v)

        kept = 0
        max_err = 0.0f0
        mismatch = 0
        @inbounds for i in 1:n_points
            base = (i - 1) * POINT_STEP + 1
            rx, ry, rz, rv = ref_project(cp[base], cp[base+1], cp[base+2])
            ref_x[i] = rx; ref_y[i] = ry; ref_z[i] = rz; ref_v[i] = rv
            if rv == Int32(1); kept += 1; end
            if h_v[i] != rv
                mismatch += 1
                continue
            end
            if rv == Int32(1)
                e = abs(h_x[i] - rx); if e > max_err; max_err = e; end
                e = abs(h_y[i] - ry); if e > max_err; max_err = e; end
                e = abs(h_z[i] - rz); if e > max_err; max_err = e; end
            end
        end
        @printf("[batch %d] kept=%d/%d max_err=%.3e valid_mismatch=%d\n",
                b, kept, n_points, max_err, mismatch)
        ok_all = ok_all && (mismatch == 0) && (max_err <= 1f-3)
    end

    @printf("Average kernel execution time: %.1f (us)\n", total_us / n_batches)
    println(ok_all ? "PASS" : "FAIL")
end

main()
