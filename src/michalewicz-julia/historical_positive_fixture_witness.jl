using CUDA,Printf,SHA
module Historical
 original=joinpath(@__DIR__,"reference_original","main.jl")
 text=read(original,String)
 # Preserve every original function; omit only the CLI invocation.
 @assert count("exit(main())",text)==1
 include_string(@__MODULE__,replace(text,"exit(main())"=>""),original)
end
n=257
for dim in [2,5,10]
 host=fill(4f0,n*dim);values=CuArray(host);out=CUDA.zeros(Float32,n)
 @cuda threads=256 blocks=cld(n,256) Historical.eval_kernel!(values,out,Int32(n),Int32(dim))
 CUDA.device_synchronize(blocking=true);got=Array(out);old_gpu_min=minimum(got);old_cpu_reference=Historical.reference(host,n,dim);old_pass=isapprox(old_gpu_min,old_cpu_reference;rtol=1f-5,atol=1f-5);witness=all(v->isfinite(v)&&v>0,got)&&old_cpu_reference==0f0&&!old_pass
 path=joinpath(ENV["STUDY_OUTPUT_DIR"],"historical_gpu_objectives_dim$(dim).bin");write(path,got)
 @printf("HISTORICAL_WITNESS {\"dim\":%d,\"n\":%d,\"old_gpu_min\":%.12g,\"old_cpu_reference\":%.12g,\"historical_selfcheck_would_pass\":%s,\"positive_fixture_counterexample\":%s,\"artifact\":\"%s\",\"sha256\":\"%s\"}\n",dim,n,old_gpu_min,old_cpu_reference,old_pass ? "true" : "false",witness ? "true" : "false",path,bytes2hex(sha256(read(path))))
 @assert witness
end
