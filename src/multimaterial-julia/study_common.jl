using CUDA, Printf
include("device_components.jl")
const INPUT_SPECS = [("imaterial", 7680, true), ("nextfrac", 17280, true), ("matids", 17280, true), ("mmc_index", 5761, true), ("mmc_i", 5760, true), ("mmc_j", 5760, true), ("rho_cell", 7680, false), ("rho_list", 17280, false), ("vf_list", 17280, false), ("t_cell", 7680, false), ("t_list", 17280, false), ("V", 7680, false), ("x", 7680, false), ("y", 7680, false), ("n", 50, false)]
const OUTPUT_SPECS = [("rho_ave", 7680), ("p_cell", 7680), ("p_list", 17280), ("neighbor_cell", 7680), ("neighbor_list", 17280)]

function study_main(mode)
 parent=@__DIR__
 raw=read(joinpath(parent,"input_mode$(mode)_f64.bin"));input=copy(reinterpret(Float64,raw))
 offset=0
 host=map(INPUT_SPECS) do (name,n,isint)
  a=input[offset+1:offset+n];offset+=n
  Symbol(name)=>(isint ? Int32.(a) : a)
 end
 dh= (;host...); d=map(CuArray,dh)
 oh=(;[Symbol(name)=>vcat(zeros(n),fill(-987654.25,17)) for (name,n) in OUTPUT_SPECS]...)
 o=map(CuArray,oh)
 expected=copy(reinterpret(Float64,read(joinpath(parent,"oracle_f64.bin"))))
 m=Val(mode)
 for phase in 1:3
  launch_phase!(phase,d,o,m);CUDA.device_synchronize(blocking=true)
 end
 for trial in 1:5
  CUDA.device_synchronize(blocking=true);t0=time_ns()
  launch_phase!(1,d,o,m);CUDA.device_synchronize(blocking=true);t1=time_ns()
  launch_phase!(2,d,o,m);CUDA.device_synchronize(blocking=true);t2=time_ns()
  launch_phase!(3,d,o,m);CUDA.device_synchronize(blocking=true);t3=time_ns()
  prefix=reduce(vcat,[Float64.(Array(v)) for v in values(d)])
  actual=reduce(vcat,[Array(v) for v in values(o)])
  errors=abs.(actual.-expected)
  good=all(isfinite,actual)&&all(errors .<= 1e-8 .+ 2e-13 .* abs.(expected))
  ma=maximum(errors);mr=maximum([expected[k]==0 ? 0.0 : errors[k]/abs(expected[k]) for k in eachindex(expected)])
  path=joinpath(ENV["STUDY_OUTPUT_DIR"],"trial$(trial).bin")
  open(path,"w") do f
   write(f,prefix);write(f,actual)
  end
  @printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"algorithm_seconds\":[%.17g,%.17g,%.17g],\"correct\":%s,\"checked_values\":%d,\"max_abs_error\":%.17g,\"max_rel_error\":%.17g,\"timing_valid\":true,\"jit_excluded\":true,\"region\":\"compact_alg1_then_alg2_then_alg3_each_device_completion\",\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%d}}\n",trial,(t3-t0)*1e-9,(t1-t0)*1e-9,(t2-t1)*1e-9,(t3-t2)*1e-9,good ? "true" : "false",length(expected),ma,mr,path,length(prefix)+length(actual))
  flush(stdout);good||exit(1)
 end
 CUDA.device_synchronize(blocking=true)
end
