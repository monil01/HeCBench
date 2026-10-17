@inline function rng_step(seed::UInt64)
 seed=(UInt64(2806196910506780709)*seed+UInt64(1))&UInt64(0x7fffffffffffffff)
 return seed,Float64(seed)/9223372036854775808.0
end
@inline function sample(id,L)
 seed=UInt64(id);seed,x=rng_step(seed);seed,y=rng_step(seed);seed,z=rng_step(seed)
 return L*x,L*y,L*z
end
@inline function factor(x,y,z,atoms,L,::Val{FIX}) where FIX
 E=0.0
 for i in 1:length(atoms)
  @inbounds a=atoms[i];dx=x-a[1];dy=y-a[2];dz=z-a[3]
  if FIX
   dx-=L*round(dx/L);dy-=L*round(dy/L);dz-=L*round(dz/L)
  else
   up=.5L;low=-up
   dx=dx>up ? dx-L : dx;dx=dx>up ? dx-L : dx
   dy=dy>up ? dy-L : dy;dy=dy<=low ? dy-L : dy
   dz=dz<=low ? dz-L : dz;dz=dz<=low ? dz-L : dz
  end
  rinv=1/sqrt(dx*dx+dy*dy+dz*dz);s=rinv*a[5];s6=s^6;E+=4a[4]*(s6*s6-s6)
 end
 return exp(-E/(8.314*298.0))
end
function study(run!)
 file=joinpath(@__DIR__,"..","henry-cuda","IRMOF-1.cssr");lines=readlines(file);L=parse(Float64,lines[1]);na=parse(Int,lines[3]);eps=Dict("Zn"=>96.152688,"O"=>66.884614,"C"=>88.480032,"H"=>57.276566);sig=Dict("Zn"=>3.095775,"O"=>3.424075,"C"=>3.580425,"H"=>3.150565)
 atoms=NTuple{5,Float64}[]
 for line in lines[5:4+na]
  fields=split(line);push!(atoms,(L*parse(Float64,fields[3]),L*parse(Float64,fields[4]),L*parse(Float64,fields[5]),eps[fields[2]],sig[fields[2]]))
 end
 n=4096;want=Float64[factor(sample(i,L)...,atoms,L,Val(false)) for i in 0:n-1];physical=Float64[factor(sample(i,L)...,atoms,L,Val(true)) for i in 0:n-1];println("STUDY_SCIENTIFIC native_boundary_vs_minimum_image_max_error=",maximum(abs.(want.-physical))," mismatched_above_1e-10=",count(>(1e-10),abs.(want.-physical))," total_points=",n)
 da=CuArray(atoms);out=CUDA.zeros(Float64,n);run!(out,da,L);CUDA.synchronize()
 for trial in 1:5
  total=0.
  for k in 1:10
   CUDA.synchronize();t=time_ns();run!(out,da,L);CUDA.synchronize();total+=(time_ns()-t)*1e-9/10
  end
  got=Array(out);err=maximum(abs.(got.-want));ok=all(isfinite,got)&&all(abs.(got.-want).<=1e-10 .+1e-9.*abs.(want));path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",n,",\"max_abs_error\":",err,",\"timing_region\":\"mean_insertion_kernel_excluding_host_reduction\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",n,",\"byte_order\":\"little\"}}")
  ok||error("Original-policy Boltzmann factor companion failed")
 end
end
