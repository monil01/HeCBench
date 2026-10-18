
using CUDA,Printf
include(joinpath(@__DIR__,"study_math.jl"));include(joinpath(@__DIR__,"study_cases.jl"))
const G=16;const GUARD=12345.5f0;const IGUARD=Int32(1234567)
function validate(a,masks)
 x=Array(a.x);data=Array(a.data);b=Array(a.bg);o=Array(a.obs);ns=Array(a.ns);ok=b==a.hb&&o==a.ho&&ns==a.hn
 ok&=all(==(GUARD),x[1:G])&&all(==(GUARD),x[end-G+1:end])&&all(==(GUARD),data[1:G])&&all(==(GUARD),data[end-G+1:end]);mo=a.mo;n=a.n;rows=Int(a.exact+a.sampled);cols=Int(a.cols);nbg=Int(a.nbg)
 @inbounds for i in 1:n;ok&=x[i+G]==Float32(masks[mo+i]);end
 @inbounds for row in 0:rows-1,donor in 0:nbg-1,col in 0:cols-1
  expected=masks[mo+row*cols+col+1]==1 ? a.ho[col+17] : a.hb[donor*cols+col+17];ok&=data[(row*nbg+donor)*cols+col+17]==expected
 end
 ok,vcat(x,data,b,o,Float32.(ns))
end
function run_study(launch)
 features=collect(reinterpret(Float32,read(joinpath(@__DIR__,"features_f32.bin"))));masks=collect(read(joinpath(@__DIR__,"oracle_masks_u8.bin")));states=Any[];prefix=Float32[]
 for(exact,sampled,cols,nbg,kmax,mo,fo,seed)in CASES
  n=(exact+sampled)*cols;dn=n*nbg;hx=vcat(fill(GUARD,G),zeros(Float32,n),fill(GUARD,G));for row in 0:exact-1;hx[row*cols+row+17]=1f0;hx[row*cols+row+18]=1f0;end
  hb=vcat(fill(GUARD,G),features[fo+1:fo+nbg*cols],fill(GUARD,G));ho=vcat(fill(GUARD,G),features[fo+nbg*cols+1:fo+(nbg+1)*cols],fill(GUARD,G));hn=vcat(fill(IGUARD,G),Int32[kmax-i%2 for i in 0:sampled÷2-1],fill(IGUARD,G));hd=fill(GUARD,dn+2G);x=CuArray(hx);data=CuArray(hd);bg=CuArray(hb);obs=CuArray(ho);ns=CuArray(hn)
  for v in(x,bg,obs,ns);append!(prefix,Float32.(Array(v)));end;append!(prefix,Array(view(data,1:G)));append!(prefix,Array(view(data,dn+G+1:dn+2G)));for i in 0:3;push!(prefix,Float32((UInt64(seed)>>(16i))&UInt64(65535)));end
  push!(states,(exact=Int32(exact),sampled=Int32(sampled),cols=Int32(cols),nbg=Int32(nbg),seed=UInt64(seed),n=n,dn=dn,mo=mo,hx=hx,hb=hb,ho=ho,hn=hn,hd=hd,x=x,data=data,bg=bg,obs=obs,ns=ns))
 end
 function reset(a);copyto!(a.x,a.hx);copyto!(a.data,a.hd);CUDA.device_synchronize();end
 for a in states;launch(a);CUDA.device_synchronize();reset(a);end
 for trial in 1:5
  artifact=copy(prefix);sizehint!(artifact,96148239);ok=true;total=0.
  for a in states
   reset(a);start=time_ns();launch(a);CUDA.device_synchronize();total+=(time_ns()-start)*1e-9;pass,values=validate(a,masks);ok&=pass;append!(artifact,values)
  end
  file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"sampling_$(trial).bin");write(file,artifact);@printf("STUDY_TRIAL {\"trial\":%d,\"correct\":%s,\"jit_excluded\":true,\"time_seconds\":%.12g,\"timing_region\":\"sum_seven_source_exact_and_concurrent_sample_scatter_pipelines_plus_completion\",\"max_abs_error\":0,\"max_rel_error\":0,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%d,\"shape\":[%d],\"byte_order\":\"little\"}}\n",trial,string(ok),total,file,length(artifact),length(artifact))
 end
end
