@inline function term(x,y,z,a,b,c,q,s,::Val{M}) where M
 dx=x-a;dy=y-b;dz=z-c;dist=sqrt(dx*dx+dy*dy+dz*dz);arg=-.125f0*(dist-s)
 e=M==2 ? Base.FastMath.exp_fast(arg) : exp(arg)
 return (q/dist)*e/(1f0+.125f0*s)
end
function study(run!)
 na=129;n=1024
 ax=Float32[Float32(mod(i,17))/64f0 for i in 0:na-1];ay=Float32[Float32(mod(i,13))/64f0 for i in 0:na-1];az=Float32[Float32(mod(i,11))/64f0 for i in 0:na-1];q=Float32[Float32(mod(i,17)+1)/1024f0 for i in 0:na-1];sz=Float32[Float32(mod(i,4)+1)/8f0 for i in 0:na-1]
 gx=Float32[1f0+Float32(mod(i,31))/64f0 for i in 0:n-1];gy=Float32[1f0+Float32(mod(7i,37))/64f0 for i in 0:n-1];gz=Float32[1f0+Float32(mod(11i,41))/64f0 for i in 0:n-1]
 want=zeros(Float32,n)
 for j in 1:n
  total=0.0
  for i in 1:na
   d=sqrt((Float64(gx[j])-ax[i])^2+(Float64(gy[j])-ay[i])^2+(Float64(gz[j])-az[i])^2)
   total+=(Float64(q[i])/d)*exp(-0.125*(d-sz[i]))/(1+0.125*sz[i])
  end
  want[j]=Float32(total)
 end
 da,db,dc,dq,ds=map(CuArray,(ax,ay,az,q,sz));dx=CuArray(collect(reinterpret(NTuple{4,Float32},gx)));dy=CuArray(collect(reinterpret(NTuple{4,Float32},gy)));dz=CuArray(collect(reinterpret(NTuple{4,Float32},gz)));out=CuArray(fill((0f0,0f0,0f0,0f0),n÷4));template=CUDA.zeros(Float32,320);args=(da,db,dc,dx,dy,dz,dq,ds,out,template)
 for m in (Val(0),Val(1),Val(2)); run!(args...,m);CUDA.synchronize();end
 for trial in 1:5
  total=0.;got=Float32[]
  for m in (Val(0),Val(1),Val(2))
   CUDA.synchronize();t=time_ns();for k in 1:10;run!(args...,m);end;CUDA.synchronize();total+=(time_ns()-t)*1e-9/10;append!(got,collect(reinterpret(Float32,Array(out))))
  end
  expected=repeat(want,3);err=maximum(abs.(got.-expected));ok=all(isfinite,got)&&all(>(0f0),got)&&err<=1f-3;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(got),",\"max_abs_error\":",err,",\"timing_region\":\"sum_three_vector_kernel_bulk_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(got),",\"byte_order\":\"little\"}}")
  ok||error("Independent Float64 screened-potential oracle failed")
 end
end
