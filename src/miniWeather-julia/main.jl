using CUDA
using Printf
using StaticArrays

const PI_D = 3.14159265358979323846264338327
const GRAV = 9.8
const CP = 1004.0
const CV = 717.0
const RD = 287.0
const P0 = 1.0e5
const C0 = 27.5629410929725921310572974482
const GAMM = 1.40027894002789400278940027894
const XLEN = 2.0e4
const ZLEN = 1.0e4
const HV_BETA = 0.25
const CFL = 1.50
const MAX_SPEED = 450.0
const HS = Int32(2)
const NUM_VARS = Int32(4)
const ID_DENS = Int32(0)
const ID_UMOM = Int32(1)
const ID_WMOM = Int32(2)
const ID_RHOT = Int32(3)
const DIR_X = Int32(1)
const DIR_Z = Int32(2)
const DATA_SPEC_THERMAL = Int32(2)
const NQPOINTS = 3
const QPOINTS = (0.112701665379258311482073460022, 0.5, 0.887298334620741688517926539980)
const QWEIGHTS = (0.277777777777777777777777777779, 0.444444444444444444444444444444, 0.277777777777777777777777777779)

@inline state_idx(ll, k, i, nx, nz, hs) = Int(ll) * Int(nz + 2hs) * Int(nx + 2hs) + Int(k) * Int(nx + 2hs) + Int(i) + 1
@inline flux_idx(ll, k, i, nx, nz) = Int(ll) * Int(nz + 1) * Int(nx + 1) + Int(k) * Int(nx + 1) + Int(i) + 1
@inline tend_idx(ll, k, i, nx, nz) = Int(ll) * Int(nz) * Int(nx) + Int(k) * Int(nx) + Int(i) + 1

function hydro_const_theta(z)
    theta0 = 300.0
    exner0 = 1.0
    t = theta0
    exner = exner0 - GRAV * z / (CP * theta0)
    p = P0 * exner^(CP / RD)
    rt = (p / C0)^(1.0 / GAMM)
    return rt / t, t
end

function sample_ellipse_cosine(x, z, amp, x0, z0, xrad, zrad)
    dist = sqrt(((x - x0) / xrad)^2 + ((z - z0) / zrad)^2) * PI_D / 2.0
    return dist <= PI_D / 2.0 ? amp * cos(dist)^2 : 0.0
end

function thermal(x, z)
    hr, ht = hydro_const_theta(z)
    return 0.0, 0.0, 0.0, sample_ellipse_cosine(x, z, 3.0, XLEN / 2.0, 2000.0, 2000.0, 2000.0), hr, ht
end

function initialize_state(nx_glob::Int32, nz_glob::Int32)
    nranks = Int32(1)
    myrank = Int32(0)
    dx = XLEN / nx_glob
    dz = ZLEN / nz_glob
    nper = Float64(nx_glob) / nranks
    i_beg = Int32(round(Int, nper * myrank))
    i_end = Int32(round(Int, nper * (myrank + 1)) - 1)
    nx = i_end - i_beg + Int32(1)
    nz = nz_glob
    k_beg = Int32(0)
    dt = min(dx, dz) / MAX_SPEED * CFL

    state = zeros(Float64, Int((nx + 2HS) * (nz + 2HS) * NUM_VARS))
    state_tmp = similar(state)
    hy_dens_cell = zeros(Float64, Int(nz + 2HS))
    hy_dens_theta_cell = zeros(Float64, Int(nz + 2HS))
    hy_dens_int = zeros(Float64, Int(nz + 1))
    hy_dens_theta_int = zeros(Float64, Int(nz + 1))
    hy_pressure_int = zeros(Float64, Int(nz + 1))

    for k in Int32(0):(nz + 2HS - 1), i in Int32(0):(nx + 2HS - 1)
        for kk in 1:NQPOINTS, ii in 1:NQPOINTS
            x = (i_beg + i - HS + 0.5) * dx + (QPOINTS[ii] - 0.5) * dx
            z = (k_beg + k - HS + 0.5) * dz + (QPOINTS[kk] - 0.5) * dz
            r, u, w, t, hr, ht = thermal(x, z)
            wt = QWEIGHTS[ii] * QWEIGHTS[kk]
            state[state_idx(ID_DENS, k, i, nx, nz, HS)] += r * wt
            state[state_idx(ID_UMOM, k, i, nx, nz, HS)] += (r + hr) * u * wt
            state[state_idx(ID_WMOM, k, i, nx, nz, HS)] += (r + hr) * w * wt
            state[state_idx(ID_RHOT, k, i, nx, nz, HS)] += ((r + hr) * (t + ht) - hr * ht) * wt
        end
        for ll in Int32(0):(NUM_VARS - 1)
            state_tmp[state_idx(ll, k, i, nx, nz, HS)] = state[state_idx(ll, k, i, nx, nz, HS)]
        end
    end

    for k in Int32(0):(nz + 2HS - 1)
        z = (k_beg + k - HS + 0.5) * dz
        for kk in 1:NQPOINTS
            _, _, _, _, hr, ht = thermal(0.0, z)
            hy_dens_cell[Int(k) + 1] += hr * QWEIGHTS[kk]
            hy_dens_theta_cell[Int(k) + 1] += hr * ht * QWEIGHTS[kk]
        end
    end

    for k in Int32(0):nz
        z = (k_beg + k) * dz
        _, _, _, _, hr, ht = thermal(0.0, z)
        hy_dens_int[Int(k) + 1] = hr
        hy_dens_theta_int[Int(k) + 1] = hr * ht
        hy_pressure_int[Int(k) + 1] = C0 * (hr * ht)^GAMM
    end

    return (; nx, nz, i_beg, k_beg, dx, dz, dt, state, state_tmp,
        hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int)
end

function set_halo_x_kernel!(state, nx::Int32, nz::Int32, hs::Int32)
    k = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    s = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    ll = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1)
    if s < hs && k < nz && ll < NUM_VARS
        state[state_idx(ll, k + hs, s, nx, nz, hs)] = state[state_idx(ll, k + hs, nx + s, nx, nz, hs)]
        state[state_idx(ll, k + hs, nx + hs + s, nx, nz, hs)] = state[state_idx(ll, k + hs, hs + s, nx, nz, hs)]
    end
    return
end

function update_state_z_kernel!(state, nx::Int32, nz::Int32, hs::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    ll = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    if i < nx + 2hs && ll < NUM_VARS
        if ll == ID_WMOM
            state[state_idx(ll, Int32(0), i, nx, nz, hs)] = 0.0
            state[state_idx(ll, Int32(1), i, nx, nz, hs)] = 0.0
            state[state_idx(ll, nz + hs, i, nx, nz, hs)] = 0.0
            state[state_idx(ll, nz + hs + Int32(1), i, nx, nz, hs)] = 0.0
        else
            state[state_idx(ll, Int32(0), i, nx, nz, hs)] = state[state_idx(ll, hs, i, nx, nz, hs)]
            state[state_idx(ll, Int32(1), i, nx, nz, hs)] = state[state_idx(ll, hs, i, nx, nz, hs)]
            state[state_idx(ll, nz + hs, i, nx, nz, hs)] = state[state_idx(ll, nz + hs - Int32(1), i, nx, nz, hs)]
            state[state_idx(ll, nz + hs + Int32(1), i, nx, nz, hs)] = state[state_idx(ll, nz + hs - Int32(1), i, nx, nz, hs)]
        end
    end
    return
end

function compute_flux_x_kernel!(state, flux, hy_dens_cell, hy_dens_theta_cell, hv_coef::Float64, nx::Int32, nz::Int32, hs::Int32)
    k = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i < nx + Int32(1) && k < nz
        vals = MVector{4, Float64}(0, 0, 0, 0)
        d3 = MVector{4, Float64}(0, 0, 0, 0)
        for ll in Int32(0):(NUM_VARS - Int32(1))
            s0 = state[state_idx(ll, k + hs, i, nx, nz, hs)]
            s1 = state[state_idx(ll, k + hs, i + Int32(1), nx, nz, hs)]
            s2 = state[state_idx(ll, k + hs, i + Int32(2), nx, nz, hs)]
            s3 = state[state_idx(ll, k + hs, i + Int32(3), nx, nz, hs)]
            vals[Int(ll) + 1] = -s0 / 12.0 + 7.0 * s1 / 12.0 + 7.0 * s2 / 12.0 - s3 / 12.0
            d3[Int(ll) + 1] = -s0 + 3.0 * s1 - 3.0 * s2 + s3
        end
        r = vals[Int(ID_DENS) + 1] + hy_dens_cell[Int(k + hs) + 1]
        u = vals[Int(ID_UMOM) + 1] / r
        w = vals[Int(ID_WMOM) + 1] / r
        t = (vals[Int(ID_RHOT) + 1] + hy_dens_theta_cell[Int(k + hs) + 1]) / r
        p = C0 * (r * t)^GAMM
        flux[flux_idx(ID_DENS, k, i, nx, nz)] = r * u - hv_coef * d3[Int(ID_DENS) + 1]
        flux[flux_idx(ID_UMOM, k, i, nx, nz)] = r * u * u + p - hv_coef * d3[Int(ID_UMOM) + 1]
        flux[flux_idx(ID_WMOM, k, i, nx, nz)] = r * u * w - hv_coef * d3[Int(ID_WMOM) + 1]
        flux[flux_idx(ID_RHOT, k, i, nx, nz)] = r * u * t - hv_coef * d3[Int(ID_RHOT) + 1]
    end
    return
end

function compute_tend_x_kernel!(flux, tend, nx::Int32, nz::Int32, dx::Float64)
    ll = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1)
    k = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i < nx && k < nz && ll < NUM_VARS
        tend[tend_idx(ll, k, i, nx, nz)] = -(flux[flux_idx(ll, k, i + Int32(1), nx, nz)] - flux[flux_idx(ll, k, i, nx, nz)]) / dx
    end
    return
end

function compute_flux_z_kernel!(state, flux, hy_dens_int, hy_pressure_int, hy_dens_theta_int, hv_coef::Float64, nx::Int32, nz::Int32, hs::Int32)
    k = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i < nx && k < nz + Int32(1)
        vals = MVector{4, Float64}(0, 0, 0, 0)
        d3 = MVector{4, Float64}(0, 0, 0, 0)
        for ll in Int32(0):(NUM_VARS - Int32(1))
            s0 = state[state_idx(ll, k, i + hs, nx, nz, hs)]
            s1 = state[state_idx(ll, k + Int32(1), i + hs, nx, nz, hs)]
            s2 = state[state_idx(ll, k + Int32(2), i + hs, nx, nz, hs)]
            s3 = state[state_idx(ll, k + Int32(3), i + hs, nx, nz, hs)]
            vals[Int(ll) + 1] = -s0 / 12.0 + 7.0 * s1 / 12.0 + 7.0 * s2 / 12.0 - s3 / 12.0
            d3[Int(ll) + 1] = -s0 + 3.0 * s1 - 3.0 * s2 + s3
        end
        r = vals[Int(ID_DENS) + 1] + hy_dens_int[Int(k) + 1]
        u = vals[Int(ID_UMOM) + 1] / r
        w = vals[Int(ID_WMOM) + 1] / r
        t = (vals[Int(ID_RHOT) + 1] + hy_dens_theta_int[Int(k) + 1]) / r
        p = C0 * (r * t)^GAMM - hy_pressure_int[Int(k) + 1]
        if k == Int32(0) || k == nz
            w = 0.0
            d3[Int(ID_DENS) + 1] = 0.0
        end
        flux[flux_idx(ID_DENS, k, i, nx, nz)] = r * w - hv_coef * d3[Int(ID_DENS) + 1]
        flux[flux_idx(ID_UMOM, k, i, nx, nz)] = r * w * u - hv_coef * d3[Int(ID_UMOM) + 1]
        flux[flux_idx(ID_WMOM, k, i, nx, nz)] = r * w * w + p - hv_coef * d3[Int(ID_WMOM) + 1]
        flux[flux_idx(ID_RHOT, k, i, nx, nz)] = r * w * t - hv_coef * d3[Int(ID_RHOT) + 1]
    end
    return
end

function compute_tend_z_kernel!(state, flux, tend, nx::Int32, nz::Int32, hs::Int32, dz::Float64)
    ll = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1)
    k = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i < nx && k < nz && ll < NUM_VARS
        v = -(flux[flux_idx(ll, k + Int32(1), i, nx, nz)] - flux[flux_idx(ll, k, i, nx, nz)]) / dz
        if ll == ID_WMOM
            v -= state[state_idx(ID_DENS, k + hs, i + hs, nx, nz, hs)] * GRAV
        end
        tend[tend_idx(ll, k, i, nx, nz)] = v
    end
    return
end

function update_fluid_state_kernel!(state_init, state_out, tend, nx::Int32, nz::Int32, hs::Int32, dt::Float64)
    ll = (blockIdx().z - Int32(1)) * blockDim().z + threadIdx().z - Int32(1)
    k = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y - Int32(1)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i < nx && k < nz && ll < NUM_VARS
        state_out[state_idx(ll, k + hs, i + hs, nx, nz, hs)] = state_init[state_idx(ll, k + hs, i + hs, nx, nz, hs)] + dt * tend[tend_idx(ll, k, i, nx, nz)]
    end
    return
end

function compute_tendencies_x!(d_state, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, nx, nz, hs, dx, dt)
    hv_coef = -HV_BETA * dx / (16.0 * dt)
    @cuda threads=(16, 16) blocks=(cld(Int(nx + 1), 16), cld(Int(nz), 16)) compute_flux_x_kernel!(d_state, d_flux, d_hy_dens_cell, d_hy_dens_theta_cell, hv_coef, nx, nz, hs)
    @cuda threads=(16, 16, 1) blocks=(cld(Int(nx), 16), cld(Int(nz), 16), Int(NUM_VARS)) compute_tend_x_kernel!(d_flux, d_tend, nx, nz, dx)
end

function compute_tendencies_z!(d_state, d_flux, d_tend, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dz, dt)
    hv_coef = -HV_BETA * dz / (16.0 * dt)
    @cuda threads=(16, 16) blocks=(cld(Int(nx), 16), cld(Int(nz + 1), 16)) compute_flux_z_kernel!(d_state, d_flux, d_hy_dens_int, d_hy_pressure_int, d_hy_dens_theta_int, hv_coef, nx, nz, hs)
    @cuda threads=(16, 16, 1) blocks=(cld(Int(nx), 16), cld(Int(nz), 16), Int(NUM_VARS)) compute_tend_z_kernel!(d_state, d_flux, d_tend, nx, nz, hs, dz)
end

function semi_discrete_step!(state_init, state_forcing, state_out, flux, tend, hy_dens_cell, hy_dens_theta_cell, hy_dens_int, hy_dens_theta_int, hy_pressure_int, nx, nz, hs, dx, dz, dt, dir)
    if dir == DIR_X
        @cuda threads=(16, 16, 1) blocks=(1, cld(Int(nz), 16), Int(NUM_VARS)) set_halo_x_kernel!(state_forcing, nx, nz, hs)
        compute_tendencies_x!(state_forcing, flux, tend, hy_dens_cell, hy_dens_theta_cell, nx, nz, hs, dx, dt)
    else
        @cuda threads=(16, 16) blocks=(cld(Int(nx + 2hs), 16), 1) update_state_z_kernel!(state_forcing, nx, nz, hs)
        compute_tendencies_z!(state_forcing, flux, tend, hy_dens_int, hy_dens_theta_int, hy_pressure_int, nx, nz, hs, dz, dt)
    end
    @cuda threads=(16, 16, 1) blocks=(cld(Int(nx), 16), cld(Int(nz), 16), Int(NUM_VARS)) update_fluid_state_kernel!(state_init, state_out, tend, nx, nz, hs, dt)
end

function perform_timestep!(d_state, d_state_tmp, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dx, dz, dt, direction_switch)
    if direction_switch
        semi_discrete_step!(d_state, d_state, d_state_tmp, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dx, dz, dt / 3.0, DIR_X)
        semi_discrete_step!(d_state, d_state_tmp, d_state_tmp, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dx, dz, dt / 2.0, DIR_X)
        semi_discrete_step!(d_state, d_state_tmp, d_state, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dx, dz, dt, DIR_X)
        semi_discrete_step!(d_state, d_state, d_state_tmp, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dx, dz, dt / 3.0, DIR_Z)
        semi_discrete_step!(d_state, d_state_tmp, d_state_tmp, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dx, dz, dt / 2.0, DIR_Z)
        semi_discrete_step!(d_state, d_state_tmp, d_state, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dx, dz, dt, DIR_Z)
    else
        semi_discrete_step!(d_state, d_state, d_state_tmp, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dx, dz, dt / 3.0, DIR_Z)
        semi_discrete_step!(d_state, d_state_tmp, d_state_tmp, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dx, dz, dt / 2.0, DIR_Z)
        semi_discrete_step!(d_state, d_state_tmp, d_state, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dx, dz, dt, DIR_Z)
        semi_discrete_step!(d_state, d_state, d_state_tmp, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dx, dz, dt / 3.0, DIR_X)
        semi_discrete_step!(d_state, d_state_tmp, d_state_tmp, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dx, dz, dt / 2.0, DIR_X)
        semi_discrete_step!(d_state, d_state_tmp, d_state, d_flux, d_tend, d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int, d_hy_pressure_int, nx, nz, hs, dx, dz, dt, DIR_X)
    end
    return !direction_switch
end

function reductions(state, hy_dens_cell, hy_dens_theta_cell, nx, nz, hs, dx, dz)
    mass = 0.0
    te = 0.0
    for k in Int32(0):(nz - 1), i in Int32(0):(nx - 1)
        r = state[state_idx(ID_DENS, k + hs, i + hs, nx, nz, hs)] + hy_dens_cell[Int(hs + k) + 1]
        u = state[state_idx(ID_UMOM, k + hs, i + hs, nx, nz, hs)] / r
        w = state[state_idx(ID_WMOM, k + hs, i + hs, nx, nz, hs)] / r
        th = (state[state_idx(ID_RHOT, k + hs, i + hs, nx, nz, hs)] + hy_dens_theta_cell[Int(hs + k) + 1]) / r
        p = C0 * (r * th)^GAMM
        temp = th / (P0 / p)^(RD / CP)
        mass += r * dx * dz
        te += (r * (u * u + w * w) + r * CV * temp) * dx * dz
    end
    return mass, te
end

function check_output(d_mass, d_te)
    if isnan(d_mass)
        println("Mass change is NaN")
        return false
    elseif abs(d_mass) > 1e-9
        println("Mass change magnitude is too large")
        return false
    elseif isnan(d_te)
        println("Total energy change is NaN")
        return false
    elseif d_te >= 0
        println("Total energy change must be negative")
        return false
    elseif abs(d_te) > 4.5e-5
        println("Total energy change magnitude is too large")
        return false
    end
    return true
end

function main()
    nx_glob = Int32(400)
    nz_glob = Int32(200)
    sim_time = 600.0

    cfg = initialize_state(nx_glob, nz_glob)
    @printf("nx_glob, nz_glob: %d %d\n", nx_glob, nz_glob)
    @printf("dx,dz: %lf %lf\n", cfg.dx, cfg.dz)
    @printf("dt: %lf\n", cfg.dt)

    d_state = CuArray(cfg.state)
    d_state_tmp = CuArray(cfg.state_tmp)
    d_hy_dens_cell = CuArray(cfg.hy_dens_cell)
    d_hy_dens_theta_cell = CuArray(cfg.hy_dens_theta_cell)
    d_hy_dens_int = CuArray(cfg.hy_dens_int)
    d_hy_dens_theta_int = CuArray(cfg.hy_dens_theta_int)
    d_hy_pressure_int = CuArray(cfg.hy_pressure_int)
    d_flux = CUDA.zeros(Float64, Int((cfg.nz + 1) * (cfg.nx + 1) * NUM_VARS))
    d_tend = CUDA.zeros(Float64, Int(cfg.nz * cfg.nx * NUM_VARS))

    mass0, te0 = reductions(cfg.state, cfg.hy_dens_cell, cfg.hy_dens_theta_cell, cfg.nx, cfg.nz, HS, cfg.dx, cfg.dz)

    etime = 0.0
    dt = cfg.dt
    direction_switch = true
    CUDA.synchronize()
    t0 = time_ns()
    while etime < sim_time
        if etime + dt > sim_time
            dt = sim_time - etime
        end
        direction_switch = perform_timestep!(d_state, d_state_tmp, d_flux, d_tend,
            d_hy_dens_cell, d_hy_dens_theta_cell, d_hy_dens_int, d_hy_dens_theta_int,
            d_hy_pressure_int, cfg.nx, cfg.nz, HS, cfg.dx, cfg.dz, dt, direction_switch)
        etime += dt
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - t0) * 1e-9
    @printf("Total main time step loop: %lf sec\n", elapsed_s)

    final_state = Array(d_state)
    mass, te = reductions(final_state, cfg.hy_dens_cell, cfg.hy_dens_theta_cell, cfg.nx, cfg.nz, HS, cfg.dx, cfg.dz)
    d_mass = (mass - mass0) / mass0
    d_te = (te - te0) / te0
    @printf("d_mass: %le\n", d_mass)
    @printf("d_te:   %le\n", d_te)
    println(check_output(d_mass, d_te) ? "PASS" : "FAIL")
end

main()
