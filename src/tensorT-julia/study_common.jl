const D=(5,7,11,3,4,5)
function study(run!)
 d1,d2,d3,d4,d5,d6=D;n=prod(D);tile=d1*d2*d3;nb=d4*d5*d6
 input=Float64[i-100 for i in 0:n-1];expected=zeros(Float64,n)
 for f in 0:d6-1,e in 0:d5-1,d in 0:d4-1,c in 0:d3-1,b in 0:d2-1,a in 0:d1-1
  original=a+d1*(b+d2*(c+d3*(d+d4*(e+d5*f))))
  target=b+d2*(c+d3*(d+d4*(f+d6*(a+d1*e))))
  expected[target+1]=input[original+1]
 end
 si=Int32[d4,d5,d6];so=Int32[d2,d3,d1];ri=1f0 ./Float32.(si);ro=1f0 ./Float32.(so);st_i=Int32[d2*d3,d2*d3*d4*d6*d1,d2*d3*d4];st_l=Int32[d1,d1*d2,1];st_g=Int32[1,d2,d2*d3*d4*d6]
 x=CuArray(input);y=CUDA.zeros(Float64,n);args=(CuArray(si),CuArray(so),CuArray(ri),CuArray(ro),CuArray(st_i),CuArray(st_l),CuArray(st_g),x,y,CUDA.zeros(Float64,5900))
 run!(args...);CUDA.synchronize()
 for trial in 1:5
  CUDA.synchronize();t=time_ns();for k in 1:10;run!(args...);end;CUDA.synchronize();seconds=(time_ns()-t)*1e-9/10
  got=Array(y);err=maximum(abs.(got.-expected));ok=all(isfinite,got)&&err==0;path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,got),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":",ok,",\"checked_values\":",n,",\"max_abs_error\":",err,",\"timing_region\":\"mean_ten_bulk_kernel_launches\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",n,",\"byte_order\":\"little\"}}")
  ok||error("Full six-dimensional transpose oracle failed")
 end
end
