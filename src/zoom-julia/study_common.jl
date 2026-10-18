function study(allocate,launch)
 cases=[];prefix=Float32[]
 for(N,C,H,W)in((2,3,17,19),(1,2,32,33)),fixture in 0:1,mode in 1:2
  zh,zw=mode==1 ? (1.5f0,2.5f0) : (0.6f0,0.9f0);Ho=floor(Int,Float32(H)*zh);Wo=floor(Int,Float32(W)*zw);hs=abs(H-Ho)÷2;ws=abs(W-Wo)÷2;he=mode==1 ? H+hs : H-Ho-hs;we=mode==1 ? W+ws : W-Wo-ws;slots=(ceil(Int,Float32(16H)/Float32(Ho))+1)*(ceil(Int,Float32(16W)/Float32(Wo))+1);x=fixture==0 ? Float32[Float32(mod(i*17,97)-48)/16f0 for i in 0:N*C*H*W-1] : ones(Float32,N*C*H*W);a=allocate(Float32,length(x));out=allocate(Float32,length(x));copyto!(a,x);ref=zeros(Float32,length(x));pitch=H*W
  if mode==1;zoom_in_ref!(x,ref,H,W,Ho,Wo,pitch,hs,he,ws,we,N*C)
  else;zoom_out_ref!(x,ref,H,W,Ho,Wo,pitch,hs,he,ws,we,N*C);edge_pad_ref!(ref,H,W,pitch,hs,ws,hs+Ho,ws+Wo,N*C);end
  append!(prefix,x);append!(prefix,Float32[N,C,H,W,Ho,Wo,pitch,hs,he,ws,we,mode,4slots,3,zh,zw]);pars=(Int32(H),Int32(W),Int32(Ho),Int32(Wo),Int64(pitch),Int32(hs),Int32(he),Int32(ws),Int32(we));push!(cases,(a,out,pars,N*C,mode,slots,ref))
 end
 function run(c)
  a,out,pars,bc,mode,slots,ref=c;CUDA.synchronize(CUDA.context());start=time_ns()
  for rep in 1:3;fill!(out,0f0);launch(a,out,pars,bc,mode,slots);end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9/3;v=Array(out);e=maximum(abs.(v.-ref));return v,elapsed,all(isfinite,v)&&e<=1e-4,e
 end
 for c in cases;run(c);end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true;err=0.;components=Float64[]
  for c in cases;v,t,pass,e=run(c);append!(out,v);elapsed+=t;ok&=pass;err=max(err,e);push!(components,t);end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_COMPONENT_SECONDS ",join(components,","));println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum8_three_reset_zoom_pipeline_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(out),",\"byte_order\":\"little\"}}");ok||error("ZoomFullOutput")
 end
end
