function study(allocate,launch)
 cases=[];prefix=Float32[]
 for sd in (2,1),k in (1,3,5,7,9)
  pars=Int32[3,5,7,k,k,k,1,1,1,sd,sd,sd,sd,sd,sd,3,5,7];vs=630;cs=(2k^3+1)*192;x=Float32[Float32(mod(i*17+3k+sd,101)-50)/16f0 for i in 0:vs-1];colref=zeros(Float32,cs);vref=copy(x);vol2col_reference!(colref,x,2,pars...);col2vol_reference!(vref,colref,2,pars...);a=allocate(Float32,vs);c=allocate(Float32,cs);append!(prefix,x);append!(prefix,Float32[2;pars;3]);push!(cases,(pars,x,colref,vref,a,c))
 end
 function run(c)
  pars,x,cref,vref,a,col=c;copyto!(a,x);fill!(col,0);CUDA.synchronize(CUDA.context());start=time_ns();for rep in 1:3;launch(1,a,col,pars);end;CUDA.synchronize(CUDA.context());tf=(time_ns()-start)*1e-9/3;hc=Array(col);CUDA.synchronize(CUDA.context());start=time_ns();for rep in 1:3;launch(2,a,col,pars);end;CUDA.synchronize(CUDA.context());tb=(time_ns()-start)*1e-9/3;hv=Array(a);return vcat(hc,hv),tf,tb,hc==cref&&hv==vref
 end
 for c in cases;run(c);end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true;comp=Float64[]
  for c in cases;v,tf,tb,pass=run(c);append!(out,v);elapsed+=tf+tb;push!(comp,tf,tb);ok&=pass;end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_COMPONENT_SECONDS ",join(comp,","));println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum20_three_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(out),",\"byte_order\":\"little\"}}");ok||error("Vol2colCompleteOracle")
 end
end
