@inline function cross(p,vj,vk)
 slope=(vk[1]-vj[1])/(vk[2]-vj[2])
 return ((vj[2]>p[2])!=(vk[2]>p[2])) && (p[1]<slope*(p[2]-vj[2])+vj[1])
end
@inline function vertices(i)
 if i<150;return (Float32(i-75)/64f0,-75f0/64f0)
 elseif i<300;return (75f0/64f0,Float32(i-225)/64f0)
 elseif i<450;return (Float32(375-i)/64f0,75f0/64f0)
 else;return (-75f0/64f0,Float32(525-i)/64f0)
 end
end
function study(run!)
 n=16384;p=NTuple{2,Float32}[(Float32(2mod(i,257)-255)/128f0,Float32(2mod(17i+3,263)-261)/128f0) for i in 0:n-1];v=NTuple{2,Float32}[vertices(i) for i in 0:599]
 want=Int32[abs(q[1])<75f0/64f0 &&abs(q[2])<75f0/64f0 for q in p];dp=CuArray(p);dv=CuArray(v);out=CUDA.zeros(Int32,n);modes=(Val(0),Val(1),Val(2),Val(4),Val(8),Val(16),Val(32),Val(64))
 for mode in modes;run!(dp,dv,out,mode);CUDA.synchronize();end
 for trial in 1:5
  total=0.;got=Int32[]
  for mode in modes
   CUDA.synchronize();t=time_ns();for k in 1:10;run!(dp,dv,out,mode);end;CUDA.synchronize();total+=(time_ns()-t)*1e-9/10;append!(got,Array(out))
  end
  expected=repeat(want,8);err=maximum(abs.(got.-expected));ok=err==0;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(got),",\"max_abs_error\":",err,",\"timing_region\":\"sum_eight_case_bulk_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"int32\",\"count\":",length(got),",\"byte_order\":\"little\"}}")
  ok||error("Analytic rectangle membership oracle failed")
 end
end
