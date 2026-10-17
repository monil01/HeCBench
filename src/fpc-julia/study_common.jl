using CUDA
include(joinpath(@__DIR__,"study_pattern_math.jl"))
const STUDY_REGION="counter_reset_and_fpc_kernel_per_internal_repeat"
include_string(@__MODULE__,replace(read(joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"),String),"kernel_launch_loop_per_internal_repeat"=>STUDY_REGION),joinpath(@__DIR__,"..","_portable_study","OwnedStudy.jl"))
function fpc_study(factory)
 masked=!isempty(ARGS)&&ARGS[1]=="fpc2";n=262144;repeats=10
 patterns=UInt64[0x00000000,0x0000007f,0xffffff80,0x0000fffe,0xffff0001,0x12340000,0x00010001,0x00aa00bb,0xabababab,0x01010101,0x12345678,0x80000000,0x80000001,0xffffffff,0x00010000,0x00000100]
 # Independent hand-enumerated expected payload bytes for each pattern; no bitstream is emitted.
 bytes=Int32[1,1,1,2,2,2,2,2,1,1,4,2,4,1,2,2]
 host=UInt64[patterns[mod(i,16)+1] for i in 0:n-1]
 for i in eachindex(patterns)
  fpc_increment(patterns[i])==bytes[i]&&fpc_increment_masks(patterns[i])==bytes[i]||error("pattern table mismatch")
 end
 expected=Int32[div(n,16)*sum(bytes)];input=CuArray(host);output=CUDA.zeros(UInt32,1);operation! = factory(input,output,n,masked)
 launch! = ()->begin;fill!(output,UInt32(0));operation!();end
 println("STUDY_PATTERN_CONTRACT words=",n," low32_patterns=16 expected_payload_bytes=",expected[1]," device_container_bytes=",8*n," source_uncompressed_word_bytes=4 observable_scalar_count=1 mode=",masked ? "fpc2" : "fpc")
 OwnedStudy.run_trials!(launch!,()->fill!(output,UInt32(0)),()->Int32.(Array(output)),expected,repeats;atol=0.,rtol=0.)
end
