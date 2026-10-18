const ROUND_CONSTANTS=UInt32[1,32898,32906,2147516416,32907,2147483649,2147516545,32777,138,136,2147516425,2147483658,2147516555,139,32905,32771,32770,128,32778,2147483658,2147516545,32896]
const NINPUT=4608
const NOUTPUT=1536
function fixture(caseid)
 return caseid==1 ? zeros(UInt32,NINPUT) : UInt32[(UInt32(i)*UInt32(2654435761))⊻UInt32(i>>3)⊻UInt32(0xa5a55a5a) for i in 0:NINPUT-1]
end
function top(state,buffer,constants)
 for k in 0:191
  @inbounds state=(state[1]⊻buffer[k*8+1],state[2]⊻buffer[k*8+2],state[3]⊻buffer[k*8+3],state[4]⊻buffer[k*8+4],state[5]⊻buffer[k*8+5],state[6]⊻buffer[k*8+6],state[7]⊻buffer[k*8+7],state[8]⊻buffer[k*8+8],state[9],state[10],state[11],state[12],state[13],state[14],state[15],state[16],state[17],state[18],state[19],state[20],state[21],state[22],state[23],state[24],state[25])
  state=permutation(state,constants)
 end
 return state
end
function expected(input)
 output=zeros(UInt32,NOUTPUT)
 for bid in 0:2,tid in 0:63
  state=leaf_state(input,ROUND_CONSTANTS,tid,bid)
  for w in 0:7;output[tid+w*64+bid*512+1]=state[w+1];end
 end
 state=ntuple(_->UInt32(0),Val(25));trace=UInt32[]
 for _ in 1:8;state=top(state,output,ROUND_CONSTANTS);append!(trace,state);end
 return vcat(output,trace)
end
function study(launch,allocate)
 cases=[];prefix=UInt32[]
 for caseid in 1:2
  input=fixture(caseid);dd=allocate(UInt32,NINPUT);dout=allocate(UInt32,NOUTPUT);dc=allocate(UInt32,22);copyto!(dc,ROUND_CONSTANTS);append!(prefix,input);append!(prefix,ROUND_CONSTANTS);append!(prefix,zeros(UInt32,25));append!(prefix,zeros(UInt32,NOUTPUT));append!(prefix,UInt32[64,3,3,8,32,32]);push!(cases,(input,dd,dout,dc,zeros(UInt32,NOUTPUT),zeros(UInt32,200),expected(input)))
 end
 function case!(c)
  input,dd,dout,dc,output,trace,want=c;CUDA.fill!(dout,UInt32(0));state=ntuple(_->UInt32(0),Val(25));CUDA.synchronize(CUDA.context());t=time_ns()
  for it in 1:8
   copyto!(dd,input);launch(dd,dout,dc);copyto!(output,dout);state=top(state,output,ROUND_CONSTANTS)
   for j in 1:25;trace[(it-1)*25+j]=state[j];end
  end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-t)*1e-9;got=vcat(output,trace);return got,elapsed,got==want
 end
 for c in cases;got,t,ok=case!(c);ok||error("WarmupTreeHashMismatch");end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true
  for c in cases;got,t,pass=case!(c);append!(out,got);elapsed+=t;ok&=pass;end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":0,\"timing_region\":\"sum_two_eight_pass_transfer_leaf_gpu_root_cpu_pipelines\",\"jit_excluded\":true,\"internal_repeats\":8,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("FullLeafRootTrajectoryMismatch")
 end
end
