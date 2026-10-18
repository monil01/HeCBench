using CUDA,Printf
function normal_study(launch)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));pos=2;cs=Any[];@assert sizeof(Point3)==12&&sizeof(Normal4)==16
 for ci in 1:5
  W,H=Int.(pre[pos:pos+1]);pos+=2;N=W*H;payload=Point3[]
  for j in 1:N
   a,b,c=UInt32.(pre[pos:pos+2]);pos+=3;push!(payload,Point3(reinterpret(Float32,a),reinterpret(Float32,b),reinterpret(Float32,c)))
  end
  h=vcat(fill(Point3(-12345f0,-12345f0,-12345f0),64),payload,fill(Point3(-12345f0,-12345f0,-12345f0),8));d=CuArray(h);o=CUDA.fill(n4(-12345,-12345,-12345,-12345),N+24);push!(cs,(W=W,H=H,N=N,h=h,d=d,o=o,p=view(d,65:N+64),out=view(o,17:N+16),ci=ci))
 end
 @assert pos==length(pre)+1
 for c in cs;launch(c);CUDA.synchronize();end
 for trial in 1:5
  art=copy(pre);go=0;ok=true;ma=0.;mr=0.;times=Float64[]
  for c in cs
   fill!(c.o,n4(-12345,-12345,-12345,-12345));CUDA.synchronize();t=time_ns();launch(c);CUDA.synchronize();push!(times,(time_ns()-t)*1e-9);a=Array(c.o);v=Float64[z.value for q in a for z in q];ref=gold[go+1:go+length(v)];go+=length(v);ok&=all(isfinite,v)&&v[1:64]==ref[1:64]&&v[end-31:end]==ref[end-31:end]
   for i in 1:c.N
    j=64+4*(i-1);x=v[j+1:j+3];r=ref[j+1:j+3];er=abs.(x.-r);ok&=v[j+4]==ref[j+4]&&(all(iszero,r) ? x==r : all(er .<= 1e-6 .+ 3e-6 .* abs.(r)));ma=max(ma,maximum(er));mr=max(mr,maximum(er./(abs.(r).+1e-300)))
    if !all(iszero,r)
     q=c.h[i+64];ok&=abs(sum(abs2,x)-1)<=1e-5&&q.x*x[1]+q.y*x[2]+q.z*x[3]<=1e-5
    end
   end
   append!(art,v);h=Array(c.d);ok&=reinterpret(UInt8,h)==reinterpret(UInt8,c.h);v=Float64.(reinterpret(UInt32,h));ref=gold[go+1:go+length(v)];go+=length(v);ok&=v==ref;append!(art,v)
  end
  @assert go==length(gold);f=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(f,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum5_complete_pointcloud_normal_estimation_launches\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s}\n",trial,sum(times),ok ? "true" : "false",go,ma,mr,f,length(art),string(times))
 end
end
