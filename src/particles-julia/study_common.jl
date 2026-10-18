using CUDA,Printf
include("study_device.jl")
function particles_study(factory)
 N=Int32(16384);C=Int32(262144);dt=.5f0;steps=isempty(ARGS) ? 1 : parse(Int,ARGS[1]);steps==1||error("Primary certified scope is one physical step; eight-step strict failure is separately retained")
 raw=read(joinpath(@__DIR__,"study_input_pos_f32.bin"));vr=read(joinpath(@__DIR__,"study_input_vel_f32.bin"));pars=read(joinpath(@__DIR__,"study_params.bin"));sizeof(ParticleParams)==104||error("parameter ABI");p=only(reinterpret(ParticleParams,pars));hp=collect(reinterpret(NTuple{4,Float32},raw));hv=collect(reinterpret(NTuple{4,Float32},vr));cpu=collect(reinterpret(Float64,read(joinpath(@__DIR__,"study_expected_step1_f64.bin"))))
 pv=Float64[getfield(p,i) for i in 1:fieldcount(ParticleParams)];prefix=vcat(Float64.(reinterpret(Float32,raw)),Float64.(reinterpret(Float32,vr)),pv,Float64[dt,steps]);length(prefix)==131100||error("prefix")
 pos=CuArray(hp);vel=CuArray(hv);rp=copy(pos);rv=copy(vel);keys=CUDA.zeros(UInt32,N);indices=similar(keys);starts=CUDA.zeros(UInt32,C);ends=similar(starts);launch=factory(pos,vel,rp,rv,keys,indices,starts,ends,p,N,C,dt)
 reset()=(copyto!(pos,hp);copyto!(vel,hv);fill!(rp,(0f0,0f0,0f0,0f0));fill!(rv,(0f0,0f0,0f0,0f0));fill!(ends,typemax(UInt32));nothing)
 reset();CUDA.device_synchronize(blocking=true);launch();CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  reset();CUDA.device_synchronize(blocking=true);t=time_ns();launch();CUDA.device_synchronize(blocking=true);seconds=(time_ns()-t)*1e-9
  po=Array(pos);ve=Array(vel);rpo=Array(rp);rve=Array(rv);hk=Array(keys);hi=Array(indices);hs=Array(starts);he=Array(ends);metadata=true;seen=falses(N);cs=fill(typemax(UInt32),C);ce=copy(cs)
  for i in 1:Int(N)
   id=Int(hi[i])+1;h=Int(hk[i])+1
   if !(1<=id<=N&&1<=h<=C)||seen[id];metadata=false;continue;end
   seen[id]=true;a=po[id];gx=floor(Int32,(a[1]-p.ox)/p.cellX);gy=floor(Int32,(a[2]-p.oy)/p.cellY);gz=floor(Int32,(a[3]-p.oz)/p.cellZ);expectedhash=UInt32((((gz&Int32(63))*Int32(64))+(gy&Int32(63)))*Int32(64)+(gx&Int32(63)))
   metadata&=hk[i]==expectedhash&&(i==1||hk[i-1]>=hk[i]);if i==1||hk[i]!=hk[i-1];cs[h]=UInt32(i-1);end;if i==N||hk[i]!=hk[i+1];ce[h]=UInt32(i);end
  end
  metadata&=cs==hs;for i in 1:Int(C);if cs[i]!=typemax(UInt32);metadata&=ce[i]==he[i];end;end
  floats=vcat(Float64.(reinterpret(Float32,po)),Float64.(reinterpret(Float32,ve)),Float64.(reinterpret(Float32,rpo)),Float64.(reinterpret(Float32,rve)));expected=vcat(cpu[1:8Int(N)],reduce(vcat,[cpu[4Int(id)+1:4Int(id)+4] for id in hi]),reduce(vcat,[cpu[8Int(N)+4Int(id)+1:8Int(N)+4Int(id)+4] for id in hi]));diff=abs.(floats.-expected);ma=maximum(diff);mr=maximum(diff./max.(abs.(expected),1e-5));physical=all(isfinite,floats)&&ma<=1e-5&&all(a->a[4]==1f0&&all(v->-1f0+p.radius-1e-5<=v<=1f0-p.radius+1e-5,a[1:3]),po)&&all(a->a[4]==0f0,ve);ok=metadata&&physical;allok&=ok
  canonend=[hs[i]==typemax(UInt32) ? typemax(UInt32) : he[i] for i in eachindex(he)];out=vcat(prefix,Float64.(reinterpret(Float32,po)),Float64.(reinterpret(Float32,ve)),Float64.(hk),Float64.(hi),Float64.(hs),Float64.(canonend),Float64.(reinterpret(Float32,rpo)),Float64.(reinterpret(Float32,rve)));length(out)==950300||error("artifact shape");path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(path,out)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"exact_metadata_pass\":%s,\"timing_region\":\"kernel_launch_loop_per_internal_repeat\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,seconds,ok ? "true" : "false",length(out),ma,mr,metadata ? "true" : "false",path,length(out))
 end
 allok||error("Full particle state mismatch");println("PASS")
end
