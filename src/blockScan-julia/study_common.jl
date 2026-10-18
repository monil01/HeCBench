using CUDA,Printf
const STUDY_GUARD=Int32(0x13572468)
const STUDY_BTS=(1024,512,256,128,64,32)
function study_main(launch!)
 input=collect(reinterpret(Int32,read(joinpath(@__DIR__,"fixture.bin"))));@assert length(input)==3072
 expected=fill(STUDY_GUARD,1041,54)
 for fixture in 0:2,alg in 0:2,shape in 0:5
  path=(fixture*3+alg)*6+shape+1;acc=Int64(0)
  for i in 1:1024;expected[8+i,path]=Int32(acc);acc+=input[fixture*1024+i];@assert typemin(Int32)<=acc<=typemax(Int32);end
  expected[1033,path]=Int32(acc)
  @assert all(Int64(expected[9+i,path])-Int64(expected[8+i,path])==input[fixture*1024+i] for i in 1:1024)
 end
 di=CuArray(input);@assert Array(di)==input;out=CUDA.fill(STUDY_GUARD,1041,54)
 inputs=[view(di,fixture*1024+1:(fixture+1)*1024) for fixture in 0:2]
 outputs=[view(out,9:1033,path) for path in 1:54]
 for fixture in 0:2,alg in 0:2,shape in 0:5;launch!(inputs[fixture+1],outputs[(fixture*3+alg)*6+shape+1],Val(STUDY_BTS[shape+1]),Val(1024÷STUDY_BTS[shape+1]));end
 CUDA.synchronize()
 for trial in 1:5
  fill!(out,STUDY_GUARD);CUDA.synchronize();t=time_ns()
  for rep in 1:3,fixture in 0:2,alg in 0:2,shape in 0:5;launch!(inputs[fixture+1],outputs[(fixture*3+alg)*6+shape+1],Val(STUDY_BTS[shape+1]),Val(1024÷STUDY_BTS[shape+1]));end
  CUDA.synchronize();sec=(time_ns()-t)*1e-9/3;got=Array(out);ok=got==expected;err=maximum(abs.(Int64.(got).-Int64.(expected)));art=vcat(input,vec(got));file=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"output_trial$(trial).bin");write(file,art)
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":56214,\"max_abs_error\":%d,\"max_rel_error\":0,\"timing_region\":\"complete54path_exclusive_scan_suite_per_repeat\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":59286,\"byte_order\":\"little\"}}\n",trial,sec,ok ? "true" : "false",err,file)
 end
end
