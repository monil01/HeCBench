using CUDA
using Printf

const WIDTH = 256
const HEIGHT = 256
const NSUBSAMPLES = 2
const NAO_SAMPLES = 8
const BLOCK_SIZE = 16
const PI_F = Float32(pi)

function vdot(ax, ay, az, bx, by, bz)
    return ax * bx + ay * by + az * bz
end

function normalize3(x, y, z)
    len = sqrt(vdot(x, y, z, x, y, z))
    if abs(len) > 1.0f-17
        return x / len, y / len, z / len
    end
    return x, y, z
end

function sphere_intersect(orgx, orgy, orgz, dirx, diry, dirz,
                          cx, cy, cz, radius, best_t, hit,
                          px, py, pz, nx, ny, nz)
    rsx = orgx - cx
    rsy = orgy - cy
    rsz = orgz - cz
    b = vdot(rsx, rsy, rsz, dirx, diry, dirz)
    c = vdot(rsx, rsy, rsz, rsx, rsy, rsz) - radius * radius
    d = b * b - c
    if d > 0f0
        t = -b - sqrt(max(d, 0f0))
        if t > 0f0 && t < best_t
            best_t = t
            hit = Int32(1)
            px = orgx + dirx * t
            py = orgy + diry * t
            pz = orgz + dirz * t
            nx, ny, nz = normalize3(px - cx, py - cy, pz - cz)
        end
    end
    return best_t, hit, px, py, pz, nx, ny, nz
end

function plane_intersect(orgx, orgy, orgz, dirx, diry, dirz,
                         ppx, ppy, ppz, pnx, pny, pnz,
                         best_t, hit, px, py, pz, nx, ny, nz)
    d = -vdot(ppx, ppy, ppz, pnx, pny, pnz)
    v = vdot(dirx, diry, dirz, pnx, pny, pnz)
    if abs(v) >= 1.0f-17
        t = -(vdot(orgx, orgy, orgz, pnx, pny, pnz) + d) / v
        if t > 0f0 && t < best_t
            best_t = t
            hit = Int32(1)
            px = orgx + dirx * t
            py = orgy + diry * t
            pz = orgz + dirz * t
            nx = pnx
            ny = pny
            nz = pnz
        end
    end
    return best_t, hit, px, py, pz, nx, ny, nz
end

function rng_next(x::UInt32)
    x ⊻= x >> UInt32(6)
    x ⊻= x << UInt32(17)
    x ⊻= x >> UInt32(9)
    return x
end

function rng_float(x::UInt32)
    x = rng_next(x)
    bits = (x & UInt32((1 << 23) - 1)) | UInt32(0x3f800000)
    return x, reinterpret(Float32, bits) - 1f0
end

function clamp_byte(v::Float32)
    vv = ifelse(isfinite(v), v, 0f0)
    vv = min(max(vv, 0f0), 1f0)
    i = Int32(vv * 255.5f0)
    return UInt8(i)
end

function render_kernel!(img)
    x0 = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    y0 = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    if y0 >= Int32(HEIGHT) || x0 >= Int32(WIDTH)
        return
    end

    rng = UInt32(y0 * Int32(WIDTH) + x0)
    s0 = 0f0
    s1 = 0f0
    s2 = 0f0

    for v in Int32(0):Int32(NSUBSAMPLES - 1)
        for u in Int32(0):Int32(NSUBSAMPLES - 1)
            pxs = (Float32(x0) + Float32(u) / Float32(NSUBSAMPLES) - Float32(WIDTH) / 2f0) / (Float32(WIDTH) / 2f0)
            pys = -(Float32(y0) + Float32(v) / Float32(NSUBSAMPLES) - Float32(HEIGHT) / 2f0) / (Float32(HEIGHT) / 2f0)
            dirx, diry, dirz = normalize3(pxs, pys, -1f0)

            best_t = 1.0f17
            hit = Int32(0)
            ipx = 0f0; ipy = 0f0; ipz = 0f0
            inx = 0f0; iny = 0f0; inz = 0f0
            best_t, hit, ipx, ipy, ipz, inx, iny, inz =
                sphere_intersect(0f0, 0f0, 0f0, dirx, diry, dirz,
                                 -2f0, 0f0, -3.5f0, 0.5f0,
                                 best_t, hit, ipx, ipy, ipz, inx, iny, inz)
            best_t, hit, ipx, ipy, ipz, inx, iny, inz =
                sphere_intersect(0f0, 0f0, 0f0, dirx, diry, dirz,
                                 -0.5f0, 0f0, -3f0, 0.5f0,
                                 best_t, hit, ipx, ipy, ipz, inx, iny, inz)
            best_t, hit, ipx, ipy, ipz, inx, iny, inz =
                sphere_intersect(0f0, 0f0, 0f0, dirx, diry, dirz,
                                 1f0, 0f0, -2.2f0, 0.5f0,
                                 best_t, hit, ipx, ipy, ipz, inx, iny, inz)
            best_t, hit, ipx, ipy, ipz, inx, iny, inz =
                plane_intersect(0f0, 0f0, 0f0, dirx, diry, dirz,
                                0f0, -0.5f0, 0f0, 0f0, 1f0, 0f0,
                                best_t, hit, ipx, ipy, ipz, inx, iny, inz)

            if hit != Int32(0)
                opx = ipx + 0.0001f0 * inx
                opy = ipy + 0.0001f0 * iny
                opz = ipz + 0.0001f0 * inz

                b2x = inx; b2y = iny; b2z = inz
                b1x = 0f0; b1y = 0f0; b1z = 0f0
                if inx < 0.6f0 && inx > -0.6f0
                    b1x = 1f0
                elseif iny < 0.6f0 && iny > -0.6f0
                    b1y = 1f0
                elseif inz < 0.6f0 && inz > -0.6f0
                    b1z = 1f0
                else
                    b1x = 1f0
                end
                b0x = b1y * b2z - b1z * b2y
                b0y = b1z * b2x - b1x * b2z
                b0z = b1x * b2y - b1y * b2x
                b0x, b0y, b0z = normalize3(b0x, b0y, b0z)
                b1x = b2y * b0z - b2z * b0y
                b1y = b2z * b0x - b2x * b0z
                b1z = b2x * b0y - b2y * b0x
                b1x, b1y, b1z = normalize3(b1x, b1y, b1z)

                occlusion = 0f0
                for _j in Int32(1):Int32(NAO_SAMPLES)
                    for _i in Int32(1):Int32(NAO_SAMPLES)
                        rng, r0 = rng_float(rng)
                        rng, r1 = rng_float(rng)
                        theta = sqrt(r0)
                        phi = 2f0 * PI_F * r1
                        lx = cos(phi) * theta
                        ly = sin(phi) * theta
                        lz = sqrt(max(1f0 - theta * theta, 0f0))
                        rx = lx * b0x + ly * b1x + lz * b2x
                        ry = lx * b0y + ly * b1y + lz * b2y
                        rz = lx * b0z + ly * b1z + lz * b2z

                        ot = 1.0f17
                        ohit = Int32(0)
                        tpx = 0f0; tpy = 0f0; tpz = 0f0
                        tnx = 0f0; tny = 0f0; tnz = 0f0
                        ot, ohit, tpx, tpy, tpz, tnx, tny, tnz =
                            sphere_intersect(opx, opy, opz, rx, ry, rz, -2f0, 0f0, -3.5f0, 0.5f0, ot, ohit, tpx, tpy, tpz, tnx, tny, tnz)
                        ot, ohit, tpx, tpy, tpz, tnx, tny, tnz =
                            sphere_intersect(opx, opy, opz, rx, ry, rz, -0.5f0, 0f0, -3f0, 0.5f0, ot, ohit, tpx, tpy, tpz, tnx, tny, tnz)
                        ot, ohit, tpx, tpy, tpz, tnx, tny, tnz =
                            sphere_intersect(opx, opy, opz, rx, ry, rz, 1f0, 0f0, -2.2f0, 0.5f0, ot, ohit, tpx, tpy, tpz, tnx, tny, tnz)
                        ot, ohit, tpx, tpy, tpz, tnx, tny, tnz =
                            plane_intersect(opx, opy, opz, rx, ry, rz, 0f0, -0.5f0, 0f0, 0f0, 1f0, 0f0, ot, ohit, tpx, tpy, tpz, tnx, tny, tnz)
                        if ohit != Int32(0)
                            occlusion += 1f0
                        end
                    end
                end
                occ = (Float32(NAO_SAMPLES * NAO_SAMPLES) - occlusion) / Float32(NAO_SAMPLES * NAO_SAMPLES)
                s0 += occ
                s1 += occ
                s2 += occ
            end
        end
    end

    inv = 1f0 / Float32(NSUBSAMPLES * NSUBSAMPLES)
    idx = Int(3 * (y0 * Int32(WIDTH) + x0)) + 1
    @inbounds begin
        img[idx] = clamp_byte(s0 * inv)
        img[idx + 1] = clamp_byte(s1 * inv)
        img[idx + 2] = clamp_byte(s2 * inv)
    end
    return
end

function saveppm(fname::String, w::Int, h::Int, img::Vector{UInt8})
    open(fname, "w") do io
        write(io, "P6\n")
        write(io, "$w $h\n")
        write(io, "255\n")
        write(io, img)
    end
end

function main()
    if length(ARGS) != 1
        println("Usage: main.jl <iterations>")
        exit(1)
    end
    loopmax = parse(Int, ARGS[1])
    d_img = CUDA.zeros(UInt8, WIDTH * HEIGHT * 3)
    total_ns = 0
    blocks = (cld(WIDTH, BLOCK_SIZE), cld(HEIGHT, BLOCK_SIZE))
    threads = (BLOCK_SIZE, BLOCK_SIZE)
    for _ in 1:loopmax
        CUDA.synchronize()
        start = time_ns()
        @cuda threads=threads blocks=blocks render_kernel!(d_img)
        CUDA.synchronize()
        total_ns += time_ns() - start
    end
    @printf("Average kernel time: %lf usec.\n", Float64(total_ns) / (1.0e3 * loopmax))
    saveppm("ao.ppm", WIDTH, HEIGHT, Array(d_img))
end

main()
