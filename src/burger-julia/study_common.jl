using CUDA
using Printf
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

function study_steps!(launch!,steps)
    for _ in 1:steps
        launch!()
    end
end
function study_main(prepare,args=ARGS)
    w=256;h=192;steps=20;dx=2.0/(w-1);dy=2.0/(h-1);nu=.01;dt=.0009*dx*dy/nu
    u0,v0,un0,vn0=initialize_fields(w,h,dx,dy);ur,vr,unr,vnr=copy(u0),copy(v0),copy(un0),copy(vn0)
    cpu_reference!(ur,vr,unr,vnr,w,h,steps,nu,dt,dx,dy);expected=vcat(ur,vr)
    u,v,un,vn=CuArray(u0),CuArray(v0),CuArray(un0),CuArray(vn0);launch! = prepare(u,v,un,vn,w,h,nu,dt,dx,dy);study_steps!(launch!,1);CUDA.synchronize();allok=true
    for trial in 1:5
        copyto!(u,u0);copyto!(v,v0);copyto!(un,un0);copyto!(vn,vn0);CUDA.synchronize();gb=Base.gc_num();t0=time_ns();study_steps!(launch!,steps);CUDA.synchronize();elapsed=(time_ns()-t0)*1e-9/steps;ga=Base.gc_num()
        got=vcat(Array(u),Array(v));err=abs.(got.-expected);ma=maximum(err);mr=maximum(err./max.(abs.(expected),eps(Float64)));ok=all(isfinite,got)&&ma<=1e-6;allok &=ok;file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"burger_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"timing_region\":\"four_kernel_timestep_per_step\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(got),ma,mr,steps,(ga.total_time-gb.total_time)*1e-9,ga.pause-gb.pause,file,length(got))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
