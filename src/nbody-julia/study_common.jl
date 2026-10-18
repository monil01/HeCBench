const ESENT=-8192.25f0
function particle_data(values)
    [StudyParticle(Tuple(Float32.(values[10i-9:10i-7])),Tuple(Float32.(values[10i-6:10i-4])),Tuple(Float32.(values[10i-3:10i-1])),Float32(values[10i])) for i in 1:length(values)÷10]
end
flatten_particles(values)=Float64.(collect(reinterpret(Float32,values)))
function study(allocate,launch)
    f=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixtures_f64.bin"))))
    oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))))
    budget=collect(reinterpret(Float64,read(joinpath(@__DIR__,"error_budget_f64.bin"))))
    nc=Int(f[1]);pos=2;cases=[]
    for _ in 1:nc
        n,steps=Int.(f[pos:pos+1]);dt,soft,g=Float32.(f[pos+2:pos+4]);pos+=5
        initial=particle_data(f[pos:pos+10n-1]);pos+=10n
        raw=allocate(StudyParticle,n+2);fill!(raw,PARTICLE_GUARD);pv=view(raw,2:n+1);copyto!(pv,initial)
        force_raw=allocate(StudyParticle,n+2);fill!(force_raw,PARTICLE_GUARD);qv=view(force_raw,2:n+1);copyto!(qv,initial)
        eraw=allocate(Float32,n+8);fill!(eraw,ESENT);ev=view(eraw,5:n+4);fill!(ev,0f0)
        push!(cases,(n=Int32(n),steps=steps,dt=dt,soft=soft,g=g,initial=initial,raw=raw,p=pv,force_raw=force_raw,q=qv,eraw=eraw,e=ev))
    end
    # Warm all source kernels and geometry before every timed series.
    for c in cases
        for _ in 1:c.steps
            for phase in 1:3;launch(phase,c.p,c.e,c.n,c.dt,c.soft,c.g);end
            CUDA.synchronize(CUDA.context())
        end
    end
    for trial in 1:5
        payload=copy(f);ok=true;total=0.;case_times=Float64[];step_times=Float64[];oi=1;maxerr=0.;maxnorm=0.
        for c in cases
            copyto!(c.p,c.initial);copyto!(c.q,c.initial);fill!(c.e,0f0);CUDA.synchronize(CUDA.context());ct=0.
            for step in 1:c.steps
                start=time_ns()
                for phase in 1:3;launch(phase,c.p,c.e,c.n,c.dt,c.soft,c.g);end
                CUDA.synchronize(CUDA.context());tm=(time_ns()-start)*1e-9;total+=tm;ct+=tm;push!(step_times,tm)
                # Perstep source core timer is closed; full scientific captures stay outside.
                phost=Array(c.raw);ehost=Array(c.eraw);ke=Float64(.5f0*ehost[5])
                ok&=phost[1]==PARTICLE_GUARD&&phost[end]==PARTICLE_GUARD&&all(==(ESENT),ehost[1:4])&&all(==(ESENT),ehost[end-3:end])
                got=vcat(flatten_particles(phost[2:end-1]),Float64.(ehost[5:end-4]),[ke])
                expected=oracle[oi:oi+length(got)-1];bnd=budget[oi:oi+length(got)-1];oi+=length(got)
                for (x,y,b) in zip(got,expected,bnd)
                    err=abs(x-y);maxerr=max(maxerr,err);maxnorm=max(maxnorm,b>0 ? err/b : err==0 ? 0. : 1e300);ok&=isfinite(x)&&err<=b
                end
                for (x,initial) in zip(phost[2:end-1],c.initial);ok&=x.mass==initial.mass&&x.acc==(0f0,0f0,0f0);end
                append!(payload,flatten_particles(phost));append!(payload,Float64.(ehost));push!(payload,ke)
            end
            push!(case_times,ct)
            # Separate deterministic first-force replay after ALL timed particle/energy captures.
            launch(1,c.q,c.e,c.n,c.dt,c.soft,c.g);CUDA.synchronize(CUDA.context());force_host=Array(c.force_raw)
            ok&=force_host[1]==PARTICLE_GUARD&&force_host[end]==PARTICLE_GUARD
            got=flatten_particles(force_host[2:end-1]);expected=oracle[oi:oi+length(got)-1];bnd=budget[oi:oi+length(got)-1];oi+=length(got)
            for (x,y,b) in zip(got,expected,bnd)
                err=abs(x-y);maxerr=max(maxerr,err);maxnorm=max(maxnorm,b>0 ? err/b : err==0 ? 0. : 1e300);ok&=isfinite(x)&&err<=b
            end
            for (x,initial) in zip(force_host[2:end-1],c.initial);ok&=x.pos==initial.pos&&x.vel==initial.vel&&x.mass==initial.mass;end
            append!(payload,flatten_particles(force_host))
        end
        path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w")
        println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",maxerr,",\"max_error_budget_ratio\":",maxnorm,",\"timing_region\":\"sum15_step_core_accel_update_serialenergy_context_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":45,\"force_capture\":\"untimed_initial_state_force_replay_after_all_timed_captures\",\"case_times_seconds\":",case_times,",\"step_times_seconds\":",step_times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
        ok||error("FullParticleTrajectoriesSerialEnergyInitialForceFP32ForwardBudgets")
    end
end
