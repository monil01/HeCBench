const MHU=( 9.9632117206253790e-01,  4.0825738726469545e-02,  6.3401613233199589e-04,  4.4158436861700431e-06,  1.1622058324043520e-08,  1.0000000000000000e+00,  4.0568375699663400e-02,  6.4216825832642788e-04,  4.2661664422410096e-06,  1.3559930396321903e-08, -1.3573468728873069e-11, -4.2594802366702580e-13,  7.6779952208246166e-15,  1.4260675804433780e-16, -2.6656212072499249e-18,)
const TAU=(1.7765862602413648e+01*0.02,  5.0010202770602419e-02*0.02, -7.8002064070783474e-04*0.02, -6.9399661775931530e-05*0.02,  1.6936588308244311e-06*0.02,  5.4629017090963798e-07*0.02, -1.3805420990037933e-08*0.02, -8.0678945216155694e-10*0.02,  1.6209833004622630e-11*0.02,  6.5130101230170358e-13*0.02, -6.9931705949674988e-15*0.02, -3.1161210504114690e-16*0.02,  5.0166191902609083e-19*0.02,  7.8608831661430381e-20*0.02,  4.3936315597226053e-22*0.02, -7.0535966258003289e-24*0.02, -9.0473475495087118e-26*0.02, -2.9878427692323621e-28*0.02,)
@inline function step_gate(old,x)
 s1=0.0
 for j in 5:-1:1;s1=MHU[j]+x*s1;end
 s2=0.0
 for j in 15:-1:6;s2=MHU[j]+x*s2;end
 mhu=s1/s2
 s1=0.0
 for j in 18:-1:1;s1=TAU[j]+x*s1;end
 return old+(mhu-old)*(1.0-exp(-s1))
end
function study(run!)
 n=4096;repeat=10
 voltage=Float64[-80.0+mod(i,81) for i in 0:n-1];initial=zeros(Float64,n)
 expected=copy(initial)
 for _ in 1:repeat;expected=step_gate.(expected,voltage);end
 dg=CuArray(initial);dv=CuArray(voltage)
 run!(dg,dv,n);CUDA.synchronize()
 for trial in 1:5
  copyto!(dg,initial);CUDA.synchronize();start=time_ns()
  for _ in 1:repeat;run!(dg,dv,n);end
  CUDA.synchronize();elapsed=(time_ns()-start)*1e-9
  got=Array(dg);err=maximum(abs.(got.-expected));correct=all(isfinite,got)&&err<=1e-6;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",correct,",\"checked_values\":",n,",\"max_abs_error\":",err,",\"timing_region\":\"total_ten_updates_from_zero_reset\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",n,",\"byte_order\":\"little\"}}")
  correct||error("CPU recurrence oracle failed")
 end
end
