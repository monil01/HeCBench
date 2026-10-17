using CUDA
using Printf
function study_reference!(particle_x, particle_y, random_x, random_y, map,
                    n_particles::Int, niterations::Int, grid_size::Int,
                    radius::Float32)
    for ii in 0:(n_particles - 1)
        px = particle_x[ii + 1]
        py = particle_y[ii + 1]
        map_base = ii * grid_size * grid_size
        for iter in 0:(niterations - 1)
            randx = random_x[iter * n_particles + ii + 1]
            randy = random_y[iter * n_particles + ii + 1]
            px += randx / 1000.0f0 - 0.0495f0
            py += randy / 1000.0f0 - 0.0495f0
            dx = px - trunc(px)
            dy = py - trunc(py)
            ix = floor(Int, px)
            iy = floor(Int, py)
            if px < grid_size && py < grid_size && px >= 0.0f0 && py >= 0.0f0
                if dx * dx + dy * dy <= radius * radius
                    map[map_base + iy * grid_size + ix + 1] += UInt64(1)
                end
            end
        end
        particle_x[ii + 1] = px
        particle_y[ii + 1] = py
    end
end


function study_main(prepare,args=ARGS)
    n=4096;steps=32;grid=21;radius=.5f0;repeat=isempty(args) ? 10 : parse(Int,args[1]);repeat>0||error("positive repeat")
    x0=Float32[Float32(i%21)+Float32(35+64*(i%4))/256f0 for i in 0:n-1];y0=Float32[Float32((i÷21)%21)+Float32(35+64*((i÷4)%4))/256f0 for i in 0:n-1]
    for i in 1:64
        x0[i]=isodd(i) ? -.25f0 : 21.25f0
    end
    rx=Float32[];ry=Float32[]
    for step in 0:steps-1,i in 0:n-1
        push!(rx,1000f0*(.0495f0+Float32((i*13+step*7)%17-8)/1024f0));push!(ry,1000f0*(.0495f0+Float32((i*37+step*11)%17-8)/1024f0))
    end
    ex,ey=copy(x0),copy(y0);em=zeros(UInt64,n*grid*grid);study_reference!(ex,ey,rx,ry,em,n,steps,grid,radius)
    dx,dy,drx,dry=CuArray(x0),CuArray(y0),CuArray(rx),CuArray(ry);dm=CUDA.zeros(UInt64,length(em));launch! = prepare(dx,dy,drx,dry,dm,n,steps,grid,radius);launch!();CUDA.synchronize();allok=true
    for trial in 1:5
        elapsed=0.0;gct=0.0;gcp=0
        for rep in 1:repeat
            copyto!(dx,x0);copyto!(dy,y0);CUDA.fill!(dm,UInt64(0));CUDA.synchronize();gb=Base.gc_num();t0=time_ns();launch!();CUDA.synchronize();elapsed+=(time_ns()-t0)*1e-9/repeat;ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
        end
        gx,gy,gm=Array(dx),Array(dy),Array(dm);poserr=max(maximum(abs.(gx.-ex)),maximum(abs.(gy.-ey)));countok=gm==em;ok=countok&&all(isfinite,gx)&&all(isfinite,gy)&&poserr<=1f-5;allok &=ok;got=vcat(Float64.(gx),Float64.(gy),Float64.(gm));file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"particle_trial$(trial).bin");write(file,got)
        @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":0,\"counts_exact\":%s,\"timing_region\":\"one_32step_particle_simulation_kernel_resets_excluded\",\"jit_excluded\":true,\"internal_repeats\":%d,\"gc_time_seconds_whole_region\":%.12g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,elapsed,string(ok),length(got),poserr,string(countok),repeat,gct,gcp,file,length(got))
    end
    println(allok ? "PASS" : "FAIL");allok ? 0 : 1
end
