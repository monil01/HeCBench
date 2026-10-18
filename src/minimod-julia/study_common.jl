using CUDA,Printf
const WAVE_REGIONS=((0,24,0,24,0,8),(0,24,0,8,8,16),(0,8,8,16,8,16),(8,16,8,16,8,16),(16,24,8,16,8,16),(0,24,16,24,8,16),(0,24,0,24,16,24))
function wave_study(launch)
 pre=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixture.bin"))));gold=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle.bin"))));@assert pre[1:7]==[3,24,40,32,32,20,8];c0=Float32(pre[8]);c=Tuple(Float32.(pre[9:20]));hd=Tuple(Float32.(pre[21:23]));pos=24;cs=Any[];active=falses(40960);pml=falses(40960)
 for i in 0:23,j in 0:23,k in 0:23;z=wi(i,j,k);active[z]=true;pml[z]=!(8<=i<16&&8<=j<16&&8<=k<16);end
 for ci in 1:3
  ns=Int(pre[pos]);pos+=1;source=Float32.(pre[pos:pos+ns]);pos+=ns+1;hs=Any[];ds=Any[];vs=Any[]
  for field in 1:5
   h=vcat(fill(-12345f0,20),Float32.(pre[pos:pos+40959]),fill(-12345f0,8));pos+=40960;d=CuArray(h);push!(hs,h);push!(ds,d);push!(vs,view(d,21:40980))
  end
  push!(cs,(ns=ns,source=source,hs=hs,ds=ds,vs=vs,c0=c0,c=c,hd=hd,ci=ci))
 end
 @assert pos==length(pre)+1
 for a in cs;launch(a);CUDA.synchronize();end
 for trial in 1:5
  art=copy(pre);go=0;ok=true;ma=0.;mr=0.;times=Float64[]
  for a in cs
   for (d,h) in zip(a.ds,a.hs);copyto!(d,h);end
   CUDA.synchronize();before=[Array(d) for d in a.ds];ok&=all(reinterpret(UInt8,x)==reinterpret(UInt8,y) for (x,y) in zip(before,a.hs));t=time_ns();launch(a);CUDA.synchronize();push!(times,(time_ns()-t)*1e-9)
   order=isodd(a.ns) ? [2,1,3,4,5] : [1,2,3,4,5];after=[Array(a.ds[i]) for i in order]
   for (field,x) in enumerate(after)
    v=Float64.(x);ref=gold[go+1:go+length(v)];go+=length(v);ok&=all(isfinite,v)&&v[1:20]==ref[1:20]&&v[end-7:end]==ref[end-7:end];er=abs.(v.-ref);at=field==3 ? 1e-10 : 2e-6;ok&=(a.ci==3 ? v==ref : all(er .<= at .+ 2e-5 .* abs.(ref)));ma=max(ma,maximum(er));mr=max(mr,maximum(er./(abs.(ref).+1e-300)))
    mask=field>=4 ? trues(40960) : field==3 ? .!pml : .!active;ok&=x[21:40980][mask]==a.hs[field][21:40980][mask];append!(art,v)
   end
   for x in before
    v=Float64.(x);ref=gold[go+1:go+length(v)];go+=length(v);ok&=v==ref;append!(art,v)
   end
  end
  @assert go==length(gold);f=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(f,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_region\":\"sum3_complete_seven_region_wave_PML_step_components\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d,\"byte_order\":\"little\"},\"path_times_seconds\":%s}\n",trial,sum(times),ok ? "true" : "false",go,ma,mr,f,length(art),string(times))
 end
end
