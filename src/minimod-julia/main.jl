using CUDA
using Printf

const R = Int32(4)
const NDIM = Int32(8)
const FMAX = Float32(15.0)
const VMAX = Float32(2000.0)
const CFL = Float32(0.45)

struct Grid
    ntaperx::Int32
    ntapery::Int32
    ntaperz::Int32
    ndampx::Int32
    ndampy::Int32
    ndampz::Int32
    nx::Int32
    ny::Int32
    nz::Int32
    ldimx::Int32
    ldimy::Int32
    ldimz::Int32
    dx::Int32
    dy::Int32
    dz::Int32
    x1::Int32
    x2::Int32
    x3::Int32
    x4::Int32
    x5::Int32
    x6::Int32
    y1::Int32
    y2::Int32
    y3::Int32
    y4::Int32
    y5::Int32
    y6::Int32
    z1::Int32
    z2::Int32
    z3::Int32
    z4::Int32
    z5::Int32
    z6::Int32
    lx::Int32
    ly::Int32
    lz::Int32
    ntx::Int32
    nty::Int32
    tsx::Int32
    tsy::Int32
end

@inline pow2(x) = x * x

function init_grid(nx::Int32, ny::Int32, nz::Int32, tsx::Int32, tsy::Int32)
    dx = Int32(20)
    dy = Int32(20)
    dz = Int32(20)
    lx = Int32(4)
    ly = Int32(4)
    lz = Int32(4)
    ntaperx = Int32(3)
    ntapery = Int32(3)
    ntaperz = Int32(3)
    ldimx = nx + Int32(4) * lx
    ldimy = ny + Int32(2) * ly
    ldimz = ((nz + Int32(2) * lz + Int32(31)) ÷ Int32(32)) * Int32(32)
    @printf("ldimx: %d, ldimy: %d, ldimz: %d\n", ldimx, ldimy, ldimz)

    lambdamax = VMAX / FMAX
    ndampx = Int32(trunc(Int, ntaperx * lambdamax / Float32(dx)))
    ndampy = Int32(trunc(Int, ntapery * lambdamax / Float32(dy)))
    ndampz = Int32(trunc(Int, ntaperz * lambdamax / Float32(dz)))

    x1 = Int32(0); x2 = ndampx; x3 = ndampx; x4 = nx - ndampx; x5 = nx - ndampx; x6 = nx
    y1 = Int32(0); y2 = ndampy; y3 = ndampy; y4 = ny - ndampy; y5 = ny - ndampy; y6 = ny
    z1 = Int32(0); z2 = ndampz; z3 = ndampz; z4 = nz - ndampz; z5 = nz - ndampz; z6 = nz
    ntx = nx ÷ tsx
    nty = ny ÷ tsy
    @printf("ndamp = %d %d %d\n", ndampx, ndampy, ndampz)
    return Grid(ntaperx, ntapery, ntaperz, ndampx, ndampy, ndampz,
                nx, ny, nz, ldimx, ldimy, ldimz, dx, dy, dz,
                x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6,
                z1, z2, z3, z4, z5, z6, lx, ly, lz, ntx, nty, tsx, tsy)
end

@inline function idx3(g::Grid, i::Int32, j::Int32, k::Int32)
    # CUDA uses zero-based logical coordinates; Julia arrays need a final +1.
    return Int(((i + g.lx) * g.ldimy + j + g.ly) * g.ldimz + k + g.lz) + 1
end

grid_length(g::Grid) = Int(g.ldimx) * Int(g.ldimy) * Int(g.ldimz)

function init_coef(dx::Int32)
    dx2 = Float32(dx * dx)
    return Float32[-205.0f0 / 72.0f0 / dx2,
                    8.0f0 / 5.0f0 / dx2,
                   -1.0f0 / 5.0f0 / dx2,
                    8.0f0 / 315.0f0 / dx2,
                   -1.0f0 / 560.0f0 / dx2]
end

function compute_dt_sch(coefx, coefy, coefz)
    ftmp = abs(coefx[1]) + abs(coefy[1]) + abs(coefz[1])
    for i in 2:5
        ftmp += 2.0f0 * abs(coefx[i]) + 2.0f0 * abs(coefy[i]) + 2.0f0 * abs(coefz[i])
    end
    return 2.0f0 * CFL / (sqrt(ftmp) * VMAX)
end

function gaussian_source(nt::Int32, dt::Float32)
    source = Vector{Float32}(undef, Int(nt))
    sigma = 0.6f0 * FMAX
    tau = 1.0f0
    scale = 8.0f0
    for it in Int32(1):nt
        t = dt * Float32(it - Int32(1))
        source[Int(it)] = -2.0f0 * scale * sigma *
            (sigma - 2.0f0 * sigma * scale * pow2(sigma * t - tau)) *
            exp(-scale * pow2(sigma * t - tau))
    end
    return source
end

function pml_profile_init!(profile, i_min::Int32, i_max::Int32, n_first::Int32, n_last::Int32, scale::Float32)
    n = i_max - i_min + Int32(1)
    shift = i_min - Int32(1)
    first_beg = Int32(1) + shift
    first_end = n_first + shift
    last_beg = n - n_last + Int32(1) + shift
    last_end = n + shift

    for i in i_min:i_max
        profile[Int(i - i_min) + 1] = 0.0f0
    end
    tmp = scale / pow2(Float32(first_end - first_beg + Int32(1)))
    for i in Int32(1):(first_end - first_beg + Int32(1))
        profile[Int(first_end - i + Int32(1) - i_min) + 1] = pow2(Float32(i)) * tmp
    end
    for i in Int32(1):(last_end - last_beg + Int32(1))
        profile[Int(last_beg + i - Int32(1) - i_min) + 1] = pow2(Float32(i)) * tmp
    end
end

function pml_profile_extend!(g::Grid, eta, etax, etay, etaz,
                             xbeg::Int32, xend::Int32, ybeg::Int32, yend::Int32,
                             zbeg::Int32, zend::Int32)
    for ix in (xbeg - Int32(1)):(xend + Int32(1))
        for iy in (ybeg - Int32(1)):(yend + Int32(1))
            for iz in (zbeg - Int32(1)):(zend + Int32(1))
                eta[idx3(g, ix, iy, iz)] = etax[Int(ix) + 1] + etay[Int(iy) + 1] + etaz[Int(iz) + 1]
            end
        end
    end
end

function init_eta!(g::Grid, dt_sch::Float32, eta)
    for i in Int32(-1):g.nx
        for j in Int32(-1):g.ny
            for k in Int32(-1):g.nz
                eta[idx3(g, i, j, k)] = 0.0f0
            end
        end
    end

    etax = Vector{Float32}(undef, Int(g.nx) + 2)
    etay = Vector{Float32}(undef, Int(g.ny) + 2)
    etaz = Vector{Float32}(undef, Int(g.nz) + 2)
    pml_profile_init!(etax, Int32(0), g.nx + Int32(1), g.ndampx, g.ndampx,
                      dt_sch * 3.0f0 * VMAX * log(1000.0f0) / (2.0f0 * Float32(g.ndampx * g.dx)))
    pml_profile_init!(etay, Int32(0), g.ny + Int32(1), g.ndampy, g.ndampy,
                      dt_sch * 3.0f0 * VMAX * log(1000.0f0) / (2.0f0 * Float32(g.ndampy * g.dy)))
    pml_profile_init!(etaz, Int32(0), g.nz + Int32(1), g.ndampz, g.ndampz,
                      dt_sch * 3.0f0 * VMAX * log(1000.0f0) / (2.0f0 * Float32(g.ndampz * g.dz)))

    pml_profile_extend!(g, eta, etax, etay, etaz, Int32(1), g.nx, Int32(1), g.ny, g.z1 + Int32(1), g.z2)
    pml_profile_extend!(g, eta, etax, etay, etaz, Int32(1), g.nx, Int32(1), g.ny, g.z5 + Int32(1), g.z6)
    pml_profile_extend!(g, eta, etax, etay, etaz, Int32(1), g.nx, g.y1 + Int32(1), g.y2, g.z3 + Int32(1), g.z4)
    pml_profile_extend!(g, eta, etax, etay, etaz, Int32(1), g.nx, g.y5 + Int32(1), g.y6, g.z3 + Int32(1), g.z4)
    pml_profile_extend!(g, eta, etax, etay, etaz, g.x1 + Int32(1), g.x2, g.y3 + Int32(1), g.y4, g.z3 + Int32(1), g.z4)
    pml_profile_extend!(g, eta, etax, etay, etaz, g.x5 + Int32(1), g.x6, g.y3 + Int32(1), g.y4, g.z3 + Int32(1), g.z4)
end

function inner_kernel!(g, x3, x4, y3, y4, z3, z4, coef0, c, u, v, vp, phi, eta)
    k = z3 + (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    j = y3 + (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    i = x3 + (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1)
    if i <= x4 - Int32(1) && j <= y4 - Int32(1) && k <= z4 - Int32(1)
        center = idx3(g, i, j, k)
        lap = coef0 * u[center] +
            c[1] * (u[idx3(g, i + Int32(1), j, k)] + u[idx3(g, i - Int32(1), j, k)]) +
            c[2] * (u[idx3(g, i, j + Int32(1), k)] + u[idx3(g, i, j - Int32(1), k)]) +
            c[3] * (u[idx3(g, i, j, k + Int32(1))] + u[idx3(g, i, j, k - Int32(1))]) +
            c[4] * (u[idx3(g, i + Int32(2), j, k)] + u[idx3(g, i - Int32(2), j, k)]) +
            c[5] * (u[idx3(g, i, j + Int32(2), k)] + u[idx3(g, i, j - Int32(2), k)]) +
            c[6] * (u[idx3(g, i, j, k + Int32(2))] + u[idx3(g, i, j, k - Int32(2))]) +
            c[7] * (u[idx3(g, i + Int32(3), j, k)] + u[idx3(g, i - Int32(3), j, k)]) +
            c[8] * (u[idx3(g, i, j + Int32(3), k)] + u[idx3(g, i, j - Int32(3), k)]) +
            c[9] * (u[idx3(g, i, j, k + Int32(3))] + u[idx3(g, i, j, k - Int32(3))]) +
            c[10] * (u[idx3(g, i + Int32(4), j, k)] + u[idx3(g, i - Int32(4), j, k)]) +
            c[11] * (u[idx3(g, i, j + Int32(4), k)] + u[idx3(g, i, j - Int32(4), k)]) +
            c[12] * (u[idx3(g, i, j, k + Int32(4))] + u[idx3(g, i, j, k - Int32(4))])
        v[center] = 2.0f0 * u[center] + vp[center] * lap - v[center]
    end
    return
end

function pml_kernel!(g, x3, x4, y3, y4, z3, z4, hdx_2, hdy_2, hdz_2, coef0, c, u, v, vp, phi, eta)
    k = z3 + (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    j = y3 + (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    i = x3 + (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1)
    if i <= x4 - Int32(1) && j <= y4 - Int32(1) && k <= z4 - Int32(1)
        center = idx3(g, i, j, k)
        lap = coef0 * u[center] +
            c[1] * (u[idx3(g, i + Int32(1), j, k)] + u[idx3(g, i - Int32(1), j, k)]) +
            c[2] * (u[idx3(g, i, j + Int32(1), k)] + u[idx3(g, i, j - Int32(1), k)]) +
            c[3] * (u[idx3(g, i, j, k + Int32(1))] + u[idx3(g, i, j, k - Int32(1))]) +
            c[4] * (u[idx3(g, i + Int32(2), j, k)] + u[idx3(g, i - Int32(2), j, k)]) +
            c[5] * (u[idx3(g, i, j + Int32(2), k)] + u[idx3(g, i, j - Int32(2), k)]) +
            c[6] * (u[idx3(g, i, j, k + Int32(2))] + u[idx3(g, i, j, k - Int32(2))]) +
            c[7] * (u[idx3(g, i + Int32(3), j, k)] + u[idx3(g, i - Int32(3), j, k)]) +
            c[8] * (u[idx3(g, i, j + Int32(3), k)] + u[idx3(g, i, j - Int32(3), k)]) +
            c[9] * (u[idx3(g, i, j, k + Int32(3))] + u[idx3(g, i, j, k - Int32(3))]) +
            c[10] * (u[idx3(g, i + Int32(4), j, k)] + u[idx3(g, i - Int32(4), j, k)]) +
            c[11] * (u[idx3(g, i, j + Int32(4), k)] + u[idx3(g, i, j - Int32(4), k)]) +
            c[12] * (u[idx3(g, i, j, k + Int32(4))] + u[idx3(g, i, j, k - Int32(4))])
        eta_c = eta[center]
        v[center] = ((2.0f0 * eta_c + 2.0f0 - eta_c * eta_c) * u[center] +
                     (vp[center] * (lap + phi[center]) - v[center])) /
                    (2.0f0 * eta_c + 1.0f0)
        phi[center] = (phi[center] -
            ((eta[idx3(g, i + Int32(1), j, k)] - eta[idx3(g, i - Int32(1), j, k)]) *
             (u[idx3(g, i + Int32(1), j, k)] - u[idx3(g, i - Int32(1), j, k)]) * hdx_2 +
             (eta[idx3(g, i, j + Int32(1), k)] - eta[idx3(g, i, j - Int32(1), k)]) *
             (u[idx3(g, i, j + Int32(1), k)] - u[idx3(g, i, j - Int32(1), k)]) * hdy_2 +
             (eta[idx3(g, i, j, k + Int32(1))] - eta[idx3(g, i, j, k - Int32(1))]) *
             (u[idx3(g, i, j, k + Int32(1))] - u[idx3(g, i, j, k - Int32(1))]) * hdz_2)) /
            (1.0f0 + eta_c)
    end
    return
end

function add_source_kernel!(u, idx::Int32, source_value::Float32)
    u[Int(idx) + 1] += source_value
    return
end

ceildiv_i32(a::Int32, b::Int32) = Int32(cld(Int(a), Int(b)))

function launch_pml!(g, xs, xe, ys, ye, zs, ze, hdx_2, hdy_2, hdz_2, coef0, c, d_u, d_v, d_vp, d_phi, d_eta)
    blocks = (ceildiv_i32(ze - zs, NDIM), ceildiv_i32(ye - ys, NDIM), ceildiv_i32(xe - xs, NDIM))
    @cuda threads=(NDIM, NDIM, NDIM) blocks=blocks pml_kernel!(g, xs, xe, ys, ye, zs, ze,
        hdx_2, hdy_2, hdz_2, coef0, c, d_u, d_v, d_vp, d_phi, d_eta)
end

function launch_inner!(g, xs, xe, ys, ye, zs, ze, coef0, c, d_u, d_v, d_vp, d_phi, d_eta)
    blocks = (ceildiv_i32(ze - zs, NDIM), ceildiv_i32(ye - ys, NDIM), ceildiv_i32(xe - xs, NDIM))
    @cuda threads=(NDIM, NDIM, NDIM) blocks=blocks inner_kernel!(g, xs, xe, ys, ye, zs, ze,
        coef0, c, d_u, d_v, d_vp, d_phi, d_eta)
end

function target!(nsteps::Int32, g::Grid, sx::Int32, sy::Int32, sz::Int32,
                 hdx_2::Float32, hdy_2::Float32, hdz_2::Float32,
                 coefx, coefy, coefz, u, v, vp, phi, eta, source)
    d_u = CUDA.zeros(Float32, grid_length(g))
    d_v = CUDA.zeros(Float32, grid_length(g))
    d_vp = CuArray(vp)
    d_phi = CuArray(phi)
    d_eta = CuArray(eta)
    c = CuArray(Float32[coefx[2], coefy[2], coefz[2],
                        coefx[3], coefy[3], coefz[3],
                        coefx[4], coefy[4], coefz[4],
                        coefx[5], coefy[5], coefz[5]])
    coef0 = coefx[1] + coefy[1] + coefz[1]
    xmin = Int32(0); xmax = g.nx
    ymin = Int32(0); ymax = g.ny

    CUDA.synchronize()
    start_time = time_ns()
    for istep in Int32(1):nsteps
        launch_pml!(g, xmin, xmax, ymin, ymax, g.z1, g.z2, hdx_2, hdy_2, hdz_2, coef0, c, d_u, d_v, d_vp, d_phi, d_eta)
        launch_pml!(g, xmin, xmax, g.y1, g.y2, g.z3, g.z4, hdx_2, hdy_2, hdz_2, coef0, c, d_u, d_v, d_vp, d_phi, d_eta)
        launch_pml!(g, g.x1, g.x2, g.y3, g.y4, g.z3, g.z4, hdx_2, hdy_2, hdz_2, coef0, c, d_u, d_v, d_vp, d_phi, d_eta)
        launch_inner!(g, g.x3, g.x4, g.y3, g.y4, g.z3, g.z4, coef0, c, d_u, d_v, d_vp, d_phi, d_eta)
        launch_pml!(g, g.x5, g.x6, g.y3, g.y4, g.z3, g.z4, hdx_2, hdy_2, hdz_2, coef0, c, d_u, d_v, d_vp, d_phi, d_eta)
        launch_pml!(g, xmin, xmax, g.y5, g.y6, g.z3, g.z4, hdx_2, hdy_2, hdz_2, coef0, c, d_u, d_v, d_vp, d_phi, d_eta)
        launch_pml!(g, xmin, xmax, ymin, ymax, g.z5, g.z6, hdx_2, hdy_2, hdz_2, coef0, c, d_u, d_v, d_vp, d_phi, d_eta)
        src_val = istep < nsteps ? source[Int(istep) + 1] : Float32(0)
        @cuda threads=1 blocks=1 add_source_kernel!(d_v, Int32(idx3(g, sx, sy, sz) - 1), src_val)
        d_u, d_v = d_v, d_u
    end
    CUDA.synchronize()
    kernel_time = (time_ns() - start_time) / 1.0e9
    copyto!(u, Array(d_u))
    return kernel_time
end

function parse_args(args)
    nx = Int32(100); ny = Int32(100); nz = Int32(100)
    tsx = Int32(10); tsy = Int32(10)
    nsteps = Int32(1); niters = Int32(1)
    disable_warm_up_iter = true
    finalio = false
    if isempty(args)
        println("Usage:\n ./main --grid N --nsteps M --finalio\nUsing default values, no IO.\n")
    end
    i = 1
    while i <= length(args)
        if args[i] == "--grid"
            i += 1
            nx = ny = nz = Int32(parse(Int, args[i]))
        elseif args[i] == "--tsize"
            i += 1
            tsx = tsy = Int32(parse(Int, args[i]))
            @printf("tsx, tsy = %d, %d\n", tsx, tsy)
        elseif args[i] == "--nsteps"
            i += 1
            nsteps = Int32(parse(Int, args[i]))
            @printf("nsteps = %d\n", nsteps)
        elseif args[i] == "--niters"
            i += 1
            niters = Int32(parse(Int, args[i]))
            @printf("niters = %d\n", niters)
        elseif args[i] == "--warm-up"
            disable_warm_up_iter = false
            println("warm up iteration is enabled")
        elseif args[i] == "--finalio"
            finalio = true
            println("writing final wavefile is enabled")
        end
        i += 1
    end
    return nx, ny, nz, tsx, tsy, nsteps, niters, disable_warm_up_iter, finalio
end

function main()
    nx, ny, nz, tsx, tsy, nsteps, niters, disable_warm_up_iter, finalio = parse_args(ARGS)
    total_kernel_time = 0.0
    total_modeling_time = 0.0
    warm_up_iter = !disable_warm_up_iter

    for _ in 1:(disable_warm_up_iter ? Int(niters) : Int(niters) + 1)
        g = init_grid(nx, ny, nz, tsx, tsy)
        @printf("grid = %d %d %d\n", g.nx, g.ny, g.nz)
        sx = nx ÷ Int32(2)
        sy = ny ÷ Int32(2)
        sz = nz ÷ Int32(2)

        u = zeros(Float32, grid_length(g))
        v = zeros(Float32, grid_length(g))
        phi = zeros(Float32, grid_length(g))
        eta = zeros(Float32, grid_length(g))
        vp = Vector{Float32}(undef, grid_length(g))

        coefx = init_coef(g.dx)
        coefy = init_coef(g.dy)
        coefz = init_coef(g.dz)
        dt_sch = compute_dt_sch(coefx, coefy, coefz)
        vp_all = pow2(2000.0f0 * dt_sch)

        for i in Int32(0):(nx - Int32(1)), j in Int32(0):(ny - Int32(1)), k in Int32(0):(nz - Int32(1))
            phi[idx3(g, i, j, k)] = 0.0f0
            vp[idx3(g, i, j, k)] = vp_all
        end
        source = gaussian_source(nsteps, dt_sch)
        init_eta!(g, dt_sch, eta)

        hdx_2 = 1.0f0 / (4.0f0 * pow2(Float32(g.dx)))
        hdy_2 = 1.0f0 / (4.0f0 * pow2(Float32(g.dy)))
        hdz_2 = 1.0f0 / (4.0f0 * pow2(Float32(g.dz)))

        start_model = time_ns()
        kernel_time = target!(nsteps, g, sx, sy, sz, hdx_2, hdy_2, hdz_2,
                              coefx, coefy, coefz, u, v, vp, phi, eta, source)
        if warm_up_iter
            kernel_time = 0.0
        end
        model_time = (time_ns() - start_model) / 1.0e9
        if !warm_up_iter
            total_modeling_time += model_time
            total_kernel_time += kernel_time
        end

        min_u = minimum(u)
        max_u = maximum(u)
        @printf("Checksum: min_u,  max_u = %f, %f\n", min_u, max_u)
        if finalio
            open(@sprintf("snapshot.it%d.n%d.raw", nsteps, g.nz), "w") do io
                write(io, u)
            end
        end
        warm_up_iter = false
    end

    @printf("Average kernel time per iteration: %g s\n", total_kernel_time / Int(niters))
    @printf("Average modeling time per iteration: %g s\n", total_modeling_time / Int(niters))
end

main()
