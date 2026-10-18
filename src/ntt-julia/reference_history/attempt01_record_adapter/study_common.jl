
using CUDA,JSON3
include(joinpath(@__DIR__,"study_device.jl"))
function run_study(launch,model)
 N=65536;G=16;R=3;inp=reinterpret(UInt64,read(joinpath(@__DIR__,"inputs_u64.bin")));oracle=reinterpret(UInt32,read(joinpath(@__DIR__,"oracle_u32.bin")))
 for trial in 0:4
  xs=[CuArray(vcat(fill(UInt64(0x123456789abcdef),G),inp[c*N+1:(c+1)*N],fill(UInt64(0x123456789abcdef),G))) for c in 0:2];ys=[CUDA.fill(UInt32(0xdeadbeef),N+2G) for _ in 1:3];before=Array.(xs);artifact=reduce(vcat,before);ok=true;times=Float64[]
  for c in 1:3
   launch(ys[c],xs[c]);CUDA.synchronize();fill!(ys[c],UInt32(0xdeadbeef));CUDA.synchronize()
   start=time_ns();for r in 1:R;launch(ys[c],xs[c]);end;CUDA.synchronize();push!(times,(time_ns()-start)*1e-9/R)
   out=Array(ys[c]);after=Array(xs[c]);ok&=after==before[c]&&out[1:G]==fill(UInt32(0xdeadbeef),G)&&out[end-G+1:end]==fill(UInt32(0xdeadbeef),G)&&out[G+1:G+N]==oracle[(c-1)*N+1:c*N];append!(artifact,UInt64.(out));append!(artifact,after)
  end
  path="ntt_$(model)_trial$(trial).bin";write(path,artifact)
  println("HECBENCH_JSON ",JSON3.write((correct=ok,time_seconds=sum(times),timing_region="sum_three_finalstage_batches_mean_per_R3_launch_and_completion",checked_values=length(artifact),input_prefix_count=3*(N+2G),numerical_outputs=[],output_artifact=(path=abspath(path),dtype="uint64",shape=[length(artifact)],byte_order="little"))))
 end
end
