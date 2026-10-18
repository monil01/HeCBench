using CUDA,Printf
function lbm_study(launch!)
 n=Int32(17);nl=4913;pop=93347;steps=20
 prefix=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert length(prefix)==108090&&length(gold)==221197
 omega=prefix[end];lid=prefix[end-1];initial=prefix[1:pop];flags_h=Int8.(prefix[pop+1:pop+nl]);dirs_h=Int8.(prefix[pop+nl+1:pop+2nl]);where_h=Int32.(prefix[pop+2nl+1:pop+3nl])
 f0all=CUDA.fill(-12345.,pop+16);f1all=CUDA.fill(-12345.,pop+16);va=CUDA.fill(-12345f0,3nl+16);ra=CUDA.fill(-12345.,nl+16)
 fa=CuArray(vcat(fill(Int8(-83),8),flags_h,fill(Int8(-83),8)));da=CuArray(vcat(fill(Int8(-83),8),dirs_h,fill(Int8(-83),8)));wa=CuArray(vcat(fill(Int32(-991827),8),where_h,fill(Int32(-991827),8)))
 f0=view(f0all,9:pop+8);f1=view(f1all,9:pop+8);vel=view(va,9:3nl+8);rho=view(ra,9:nl+8);flags=view(fa,9:nl+8);dirs=view(da,9:nl+8);where=view(wa,9:nl+8)
 function reset!()
  copyto!(f0,initial);copyto!(f1,initial);fill!(vel,0f0);fill!(rho,1.);CUDA.synchronize()
 end
 function run!()
  a=f0;b=f1
  for step in 1:steps
   launch!(a,b,vel,flags,dirs,where,n,omega,lid);a,b=b,a;CUDA.synchronize()
  end
 end
 reset!();run!()
 for trial in 1:5
  reset!();before=vcat(Array(f0),Float64.(Array(flags)),Float64.(Array(dirs)),Float64.(Array(where)),prefix[end-3:end]);ok=reinterpret(UInt64,before)==reinterpret(UInt64,prefix)
  t=time_ns();run!();elapsed=(time_ns()-t)*1e-9
  got=vcat(Array(f0all),Array(f1all),Float64.(Array(va)),Array(ra),Float64.(Array(fa)),Float64.(Array(da)),Float64.(Array(wa)))
  err=maximum(abs.(got.-gold));rel=maximum(abs.(got.-gold)./(abs.(gold).+1e-300));ok&=all(isfinite,got)&&all(abs.(got.-gold).<=1e-7 .+ 1e-6 .*abs.(gold))
  off=0
  for (group,count) in enumerate((pop,pop,3nl,nl,nl,nl,nl))
   ok&=got[off+1:off+8]==gold[off+1:off+8]&&got[off+count+9:off+count+16]==gold[off+count+9:off+count+16]
   if group>=4;ok&=got[off+1:off+count+16]==gold[off+1:off+count+16];end
   off+=count+16
  end
  mass=0.;active=0
  for i in 1:nl
   if flags_h[i]<4;active+=1;for q in 0:18;mass+=got[8+q*nl+i];end;end
  end
  drift=abs(mass-active)/active;ok&=drift<1e-6&&minimum(got[9:pop+8])>0
  art=vcat(before,got);file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":221197,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"active_mass\":%.17g,\"active_mass_relative_drift\":%.17g,\"timing_region\":\"20_cavity_D3Q19_stream_BGK_steps_swap_sync_eachstep\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":329287,\"byte_order\":\"little\"}}\n",trial,elapsed,ok ? "true" : "false",err,rel,mass,drift,file)
 end
end
