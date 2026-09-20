using CUDA
using Printf

@inline function idx0(i::Int32, j::Int32, x_points::Int32)
    return i * x_points + j
end

function core_kernel!(u_new, v_new, u, v,
                      x_points::Int32, y_points::Int32,
                      nu::Float64, del_t::Float64,
                      del_x::Float64, del_y::Float64)
    i = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    j = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x

    if j < x_points && i < y_points
        p = idx0(i, j, x_points) + Int32(1)
        p_l = idx0(i, j - Int32(1), x_points) + Int32(1)
        p_r = idx0(i, j + Int32(1), x_points) + Int32(1)
        p_u = idx0(i - Int32(1), j, x_points) + Int32(1)
        p_d = idx0(i + Int32(1), j, x_points) + Int32(1)

        @inbounds begin
            uij = u[p]
            vij = v[p]
            u_new[p] = uij +
                (nu * del_t / (del_x * del_x)) * (u[p_r] + u[p_l] - 2.0 * uij) +
                (nu * del_t / (del_y * del_y)) * (u[p_d] + u[p_u] - 2.0 * uij) -
                (del_t / del_x) * uij * (uij - u[p_l]) -
                (del_t / del_y) * vij * (uij - u[p_u])

            v_new[p] = vij +
                (nu * del_t / (del_x * del_x)) * (v[p_r] + v[p_l] - 2.0 * vij) +
                (nu * del_t / (del_y * del_y)) * (v[p_d] + v[p_u] - 2.0 * vij) -
                (del_t / del_x) * uij * (vij - v[p_l]) -
                (del_t / del_y) * vij * (vij - v[p_u])
        end
    end
    return
end

function bound_h_kernel!(u_new, v_new, x_points::Int32, y_points::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if i < x_points
        top = idx0(Int32(0), i, x_points) + Int32(1)
        bottom = idx0(y_points - Int32(1), i, x_points) + Int32(1)
        @inbounds begin
            u_new[top] = 1.0
            v_new[top] = 1.0
            u_new[bottom] = 1.0
            v_new[bottom] = 1.0
        end
    end
    return
end

function bound_v_kernel!(u_new, v_new, x_points::Int32, y_points::Int32)
    j = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x - Int32(1)
    if j < y_points
        left = idx0(j, Int32(0), x_points) + Int32(1)
        right = idx0(j, x_points - Int32(1), x_points) + Int32(1)
        @inbounds begin
            u_new[left] = 1.0
            v_new[left] = 1.0
            u_new[right] = 1.0
            v_new[right] = 1.0
        end
    end
    return
end

function update_kernel!(u, v, u_new, v_new, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds begin
            u[i] = u_new[i]
            v[i] = v_new[i]
        end
    end
    return
end

function initialize_fields(x_points::Int, y_points::Int, del_x::Float64, del_y::Float64)
    n = x_points * y_points
    u = fill(1.0, n)
    v = fill(1.0, n)
    u_new = fill(1.0, n)
    v_new = fill(1.0, n)

    for i in 0:y_points-1
        y = i * del_y
        for j in 0:x_points-1
            x = j * del_x
            if x > 0.5 && x < 1.0 && y > 0.5 && y < 1.0
                k = i * x_points + j + 1
                u[k] = 2.0
                v[k] = 2.0
                u_new[k] = 2.0
                v_new[k] = 2.0
            end
        end
    end

    return u, v, u_new, v_new
end

function cpu_reference!(u, v, u_new, v_new, x_points::Int, y_points::Int,
                        num_itrs::Int, nu::Float64, del_t::Float64,
                        del_x::Float64, del_y::Float64)
    for _ in 1:num_itrs
        for i in 1:y_points-2
            for j in 1:x_points-2
                p = i * x_points + j + 1
                l = i * x_points + j
                r = i * x_points + j + 2
                up = (i - 1) * x_points + j + 1
                dn = (i + 1) * x_points + j + 1
                uij = u[p]
                vij = v[p]
                u_new[p] = uij +
                    (nu * del_t / (del_x * del_x)) * (u[r] + u[l] - 2.0 * uij) +
                    (nu * del_t / (del_y * del_y)) * (u[dn] + u[up] - 2.0 * uij) -
                    (del_t / del_x) * uij * (uij - u[l]) -
                    (del_t / del_y) * vij * (uij - u[up])

                v_new[p] = vij +
                    (nu * del_t / (del_x * del_x)) * (v[r] + v[l] - 2.0 * vij) +
                    (nu * del_t / (del_y * del_y)) * (v[dn] + v[up] - 2.0 * vij) -
                    (del_t / del_x) * uij * (vij - v[l]) -
                    (del_t / del_y) * vij * (vij - v[up])
            end
        end

        for j in 0:x_points-1
            top = j + 1
            bottom = (y_points - 1) * x_points + j + 1
            u_new[top] = 1.0
            v_new[top] = 1.0
            u_new[bottom] = 1.0
            v_new[bottom] = 1.0
        end

        for i in 0:y_points-1
            left = i * x_points + 1
            right = i * x_points + x_points
            u_new[left] = 1.0
            v_new[left] = 1.0
            u_new[right] = 1.0
            v_new[right] = 1.0
        end

        copyto!(u, u_new)
        copyto!(v, v_new)
    end
end

function main(args)
    if length(args) != 3
        println("Usage: main.jl <dim_x> <dim_y> <nt>")
        println("dim_x: number of grid points in the x axis")
        println("dim_y: number of grid points in the y axis")
        println("nt: number of time steps")
        return 1
    end

    x_points = parse(Int, args[1])
    y_points = parse(Int, args[2])
    num_itrs = parse(Int, args[3])
    x_len = 2.0
    y_len = 2.0
    del_x = x_len / (x_points - 1)
    del_y = y_len / (y_points - 1)
    nu = 0.01
    sigma = 0.0009
    del_t = sigma * del_x * del_y / nu

    println("2D Burger's equation")
    @printf("Grid dimension: x = %d y = %d\n", x_points, y_points)
    @printf("Number of time steps: %d\n", num_itrs)

    u, v, u_new, v_new = initialize_fields(x_points, y_points, del_x, del_y)
    d_u = CuArray(u)
    d_v = CuArray(v)
    d_u_new = CuArray(u_new)
    d_v_new = CuArray(v_new)

    grid = (cld(x_points - 2, 16), cld(y_points - 2, 16))
    block = (16, 16)
    blocks_x = cld(x_points, 256)
    blocks_y = cld(y_points, 256)
    blocks_all = cld(x_points * y_points, 256)

    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:num_itrs
        @cuda threads=block blocks=grid core_kernel!(
            d_u_new, d_v_new, d_u, d_v, Int32(x_points), Int32(y_points),
            nu, del_t, del_x, del_y)
        @cuda threads=256 blocks=blocks_x bound_h_kernel!(
            d_u_new, d_v_new, Int32(x_points), Int32(y_points))
        @cuda threads=256 blocks=blocks_y bound_v_kernel!(
            d_u_new, d_v_new, Int32(x_points), Int32(y_points))
        @cuda threads=256 blocks=blocks_all update_kernel!(
            d_u, d_v, d_u_new, d_v_new, Int32(x_points * y_points))
    end
    CUDA.synchronize()
    elapsed_s = (time_ns() - t0) * 1.0e-9
    @printf("Total kernel execution time %f (s)\n", elapsed_s)

    du = Array(d_u)
    dv = Array(d_v)

    println("Serial computing for verification...")
    ref_u, ref_v, ref_u_new, ref_v_new = initialize_fields(x_points, y_points, del_x, del_y)
    cpu_reference!(ref_u, ref_v, ref_u_new, ref_v_new,
                   x_points, y_points, num_itrs, nu, del_t, del_x, del_y)

    ok = all(abs.(du .- ref_u) .<= 1.0e-6) && all(abs.(dv .- ref_v) .<= 1.0e-6)
    println(ok ? "PASS" : "FAIL")
    return ok ? 0 : 1
end

exit(main(ARGS))
