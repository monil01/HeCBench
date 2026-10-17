using CUDA,Printf
include(joinpath(@__DIR__,"packed_accessor.jl"))
const GUARD=-123.25f0
function fixture(which)
 nr,nc,rs,cs,vs,os=which==1 ? (257,131,131,1,1,1) : which==2 ? (513,257,257,1,1,1) : (129,65,137,2,3,2)
 m=zeros(Float32,nr*nc);v=zeros(Float32,nc)
 if which==1
  ccall(:srand,Cvoid,(Cuint,),123)
  for i in eachindex(m);m[i]=Float32(ccall(:rand,Cint,()))/Float32(typemax(Cint));end
  for i in eachindex(v);v[i]=Float32(ccall(:rand,Cint,()))/Float32(typemax(Cint));end
 else
  for i in 0:nr-1,j in 0:nc-1;m[i*nc+j+1]=which==2 ? Float32((i*17+j*13)%65-32)/16f0 : Float32((i*31+j*7)%97-48)/8f0;end
  for j in 0:nc-1;v[j+1]=which==2 ? Float32((isodd(j) ? -1 : 1)*(j%9+1))/8f0 : Float32((j*5)%17-8)/4f0;end
 end
 pm=fill(GUARD,nr*rs);pv=fill(GUARD,nc*vs)
 for i in 0:nr-1,j in 0:nc-1;pm[i*rs+j*cs+1]=m[i*nc+j+1];end
 for j in 0:nc-1;pv[j*vs+1]=v[j+1];end
 gold=Float64[sum(Float64(m[i*nc+j+1])*Float64(v[j+1]) for j in 0:nc-1) for i in 0:nr-1]
 (nr,nc,rs,cs,vs,os,m,v,pm,pv,gold)
end
function study_main(launch!)
 rep=isempty(ARGS) ? 20 : parse(Int,ARGS[1]);fs=[fixture(i) for i in 1:3];ds=[];prefix=Float32[]
 for f in fs
  nr,nc,rs,cs,vs,os,m,v,pm,pv,gold=f;dm,dv=CuArray(m),CuArray(v);dpm=rs==nc&&cs==1 ? dm : CuArray(pm);dpv=vs==1 ? dv : CuArray(pv);dr=CUDA.fill(GUARD,nr);dpr=CUDA.fill(GUARD,nr*os)
  ra=StudyPacked(dpr,(Int64(nr),),(Int64(os),));ma=StudyPacked(dpm,(Int64(nr),Int64(nc)),(Int64(rs),Int64(cs)));va=StudyPacked(dpv,(Int64(nc),),(Int64(vs),));push!(ds,(dm,dv,dr,dpr,ra,ma,va))
  append!(prefix,Float32[nr,nc,rs,cs,vs,os,length(pm),length(pv),nr*os,GUARD]);append!(prefix,m);append!(prefix,v);append!(prefix,pm);append!(prefix,pv)
 end
 println("PACKED_DEVICE_ABI_BYTES ",sizeof(CUDA.cudaconvert(ds[1][5]))," ",sizeof(CUDA.cudaconvert(ds[1][6]))," ",sizeof(CUDA.cudaconvert(ds[1][7]))," isbits=",isbitstype(typeof(CUDA.cudaconvert(ds[1][6]))))
 for (f,d) in zip(fs,ds),mode in 1:2;launch!(mode,f,d);end;CUDA.synchronize()
 for trial in 1:5
  for d in ds;CUDA.fill!(d[3],GUARD);CUDA.fill!(d[4],GUARD);end;CUDA.synchronize();times=Float64[];gct=0.0;gcp=0
  for (f,d) in zip(fs,ds),mode in 1:2
   CUDA.synchronize();gb=Base.gc_num();st=time_ns();for _ in 1:rep;launch!(mode,f,d);end;CUDA.synchronize();push!(times,(time_ns()-st)*1e-9/rep);ga=Base.gc_num();gct+=(ga.total_time-gb.total_time)*1e-9;gcp+=ga.pause-gb.pause
  end
  got=Float32[];ok=true;ma=0.0;mr=0.0;guards=0
  for (f,d) in zip(fs,ds),mode in 1:2
   a=Array(mode==1 ? d[3] : d[4]);stride=mode==1 ? 1 : f[6]
   for i in eachindex(a)
    if (i-1)%stride==0
     gold=f[11][(i-1)÷stride+1];err=abs(Float64(a[i])-gold);ma=max(ma,err);mr=max(mr,err/max(abs(gold),1e-12));ok&=isfinite(a[i])&&err<=1e-3
    else;guards+=1;ok&=a[i]==GUARD;end
   end
   append!(got,a)
  end
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");art=vcat(prefix,got);write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"padding_guard_count\":%d,\"timing_region\":\"sum6paths_raw_packed_Float32_matvec_kernelonly\",\"jit_excluded\":true,\"internal_repeats\":%d,\"per_path_times_seconds\":[%s],\"gc_time_seconds_whole_region\":%.17g,\"gc_pauses\":%d,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"byte_order\":\"little\"}}\n",trial,sum(times),ok ? "true" : "false",length(got),ma,mr,guards,rep,join(times,","),gct,gcp,file,length(art));ok||error("completeFloat64dotproductsandstridedresultguards")
 end
end
