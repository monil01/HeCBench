using CUDA
using JACC
include(joinpath(@__DIR__,"..","study_common.jl"))
@inline function idx0(i::Int32, j::Int32, x_points::Int32)
    return i * x_points + j
end

function core_kernel!(ix,iy,u_new, v_new, u, v,
                      x_points::Int32, y_points::Int32,
                      nu::Float64, del_t::Float64,
                      del_x::Float64, del_y::Float64)
    i=Int32(iy);j=Int32(ix)

    if j < x_points - Int32(1) && i < y_points - Int32(1)
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
    nothing
end

function bound_h_kernel!(ix,u_new, v_new, x_points::Int32, y_points::Int32)
    i=Int32(ix)-Int32(1)

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
    nothing
end

function bound_v_kernel!(ix,u_new, v_new, x_points::Int32, y_points::Int32)
    j=Int32(ix)-Int32(1)

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
    nothing
end

function update_kernel!(ix,u, v, u_new, v_new, n::Int32)
    i=Int32(ix)

    if i <= n
        @inbounds begin
            u[i] = u_new[i]
            v[i] = v_new[i]
        end
    end
    nothing
end

function study_prepare(u,v,un,vn,w,h,nu,dt,dx,dy)
    backend=JACC.get_backend(:cuda);ST=JACC.LaunchSpec{typeof(backend)};cs=ST(;threads=(16,16),blocks=(cld(w-2,16),cld(h-2,16)),shmem_size=0,sync=false);hs=ST(;threads=256,blocks=cld(w,256),shmem_size=0,sync=false);vs=ST(;threads=256,blocks=cld(h,256),shmem_size=0,sync=false);us=ST(;threads=256,blocks=cld(w*h,256),shmem_size=0,sync=false)
    function launch!()
        JACC.parallel_for(core_kernel!,cs,(cld(w-2,16)*16,cld(h-2,16)*16),un,vn,u,v,Int32(w),Int32(h),nu,dt,dx,dy)
        JACC.parallel_for(bound_h_kernel!,hs,cld(w,256)*256,un,vn,Int32(w),Int32(h));JACC.parallel_for(bound_v_kernel!,vs,cld(h,256)*256,un,vn,Int32(w),Int32(h));JACC.parallel_for(update_kernel!,us,w*h,u,v,un,vn,Int32(w*h))
    end
end
exit(study_main(study_prepare))
