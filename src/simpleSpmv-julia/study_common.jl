using CUDA,Printf
function spmv_study(factory)
 io=open(joinpath(@__DIR__,"fixture.bin"));nc=Int(read(io,UInt32));cases=[];prefix=Float64[]
 for z in 1:nc
  n=Int(read(io,UInt32));nz=Int(read(io,UInt32));tpr=Int(read(io,UInt32));mat=CuArray(read!(io,Vector{Float32}(undef,n*n)));x=CuArray(read!(io,Vector{Float32}(undef,n)));row=CuArray(read!(io,Vector{UInt64}(undef,n+1)));col=CuArray(read!(io,Vector{UInt64}(undef,nz)));val=CuArray(read!(io,Vector{Float32}(undef,nz)))
  for a in (mat,x,row,col,val);append!(prefix,Float64.(Array(a)));end
  push!(cases,(n=Int32(n),nz=nz,tpr=tpr,mat=mat,x=x,row=row,col=col,val=val))
 end
 close(io);expected=reinterpret(Float64,read(joinpath(@__DIR__,"expected_f64.bin")));outs=[];launches=[]
 for c in cases,bs in (32,64,128,256,512,1024),method in 0:2
  out=CuArray(fill(16777215f0,Int(c.n)+16));launch=factory(c,bs,method,out);push!(outs,out);push!(launches,launch);launch()
 end
 CUDA.device_synchronize(blocking=true);allok=true
 for trial in 1:5
  times=Float64[];data=copy(prefix)
  for (launch,out)in zip(launches,outs)
   CUDA.device_synchronize(blocking=true);start=time_ns();for r in 1:100;launch();end;CUDA.device_synchronize(blocking=true);push!(times,(time_ns()-start)*1e-9/100);append!(data,Float64.(Array(out)))
  end
  observed=@view data[length(prefix)+1:end];good=isfinite.(observed).&(abs.(observed.-expected).<=1e-5.+1e-5.*abs.(expected));ok=all(good);allok&=ok;err=maximum(abs.(observed.-expected));file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"simpleSpmv_trial$(trial).bin");write(file,data)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%d,\"mismatches\":%d,\"max_abs_error\":%.12g,\"max_rel_error\":null,\"jit_excluded\":true,\"component_seconds\":[%s],\"timing_region\":\"sum90_source100repeat_kernelmeans_devicecompletion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),string(ok),length(expected),count(!,good),err,join(times,","),file,length(data))
 end
 println(allok ? "PASS" : "FAIL");allok||error("oraclefailed")
end
