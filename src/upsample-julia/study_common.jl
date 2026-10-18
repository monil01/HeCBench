const SHAPES=((2,3,7,11),(1,2,32,33))
const VARIANTS=((1,32),(1,64),(1,128),(1,256),(1,512),(1,1024),(2,8),(2,16),(2,32),(3,32),(3,64),(3,128),(3,256),(3,512),(3,1024),(4,8),(4,16),(4,32))
function fixture(B,C,H,W)
 S=B*C*H*W;x=Float32[Float32(mod(i*13,67)-33)/16f0 for i in 0:S-1];g=Float32[Float32(mod(i*17,71)-35)/16f0 for i in 0:4S-1];return x,g
end
function oracle(x,g,B,C,H,W)
 S=length(x);y=zeros(Float32,4S);dx=zeros(Float32,S)
 for bc in 0:B*C-1,h in 0:H-1,w in 0:W-1
  i=bc*H*W+h*W+w+1;base=bc*4H*W+4h*W+2w+1
  for dh in 0:1,dw in 0:1;y[base+dh*2W+dw]=x[i];end
  dx[i]=g[base]+g[base+1]+g[base+2W]+g[base+2W+1]
 end
 return y,dx
end
function study(allocate,launch)
 prefix=Float32[];cases=[]
 for(B,C,H,W)in SHAPES
  x,g=fixture(B,C,H,W);S=length(x);a=allocate(Float32,S);dg=allocate(Float32,4S);o=allocate(Float32,4S);dx=allocate(Float32,S);copyto!(a,x);copyto!(dg,g);append!(prefix,x);append!(prefix,g);append!(prefix,Float32[B,C,H,W,3]);y,refdx=oracle(x,g,B,C,H,W);push!(cases,(a,dg,o,dx,B,C,H,W,y,refdx))
 end
 function run(c,k,block)
  a,g,o,dx,B,C,H,W,y,refdx=c;target=k<=2 ? o : dx;fill!(target,-999f0);CUDA.synchronize(CUDA.context());start=time_ns();for rep in 1:3;launch(k,block,a,g,o,dx,B,C,H,W);end;CUDA.synchronize(CUDA.context());t=(time_ns()-start)*1e-9/3;out=Array(target);ref=k<=2 ? y : refdx;return out,t,out==ref
 end
 for c in cases,(k,block)in VARIANTS;run(c,k,block);end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true;components=Float64[]
  for c in cases,(k,block)in VARIANTS;v,t,pass=run(c,k,block);append!(out,v);elapsed+=t;push!(components,t);ok&=pass;end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_COMPONENT_SECONDS ",join(components,","));println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum36_three_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":",length(out),",\"byte_order\":\"little\"}}");ok||error("UpsampleCompleteOracle")
 end
end
