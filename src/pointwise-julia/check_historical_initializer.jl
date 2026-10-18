# CPU-only source witness; no CUDA import, GPU launch, or timing series.
using SHA,Printf
path=joinpath(@__DIR__,"reference_original","main.jl")
source=read(path,String)
a=findfirst("function lcg_value",source).start
b=findfirst("function init_ref",source).start
body=source[a:prevind(source,b)]
include_string(Main,body,path)
n=257
old=Float32[lcg_value(i,n) for i in 0:n-1]
new=Float32[]
for i in 0:n-1
 seed=UInt32(i)⊻UInt32(n)
 seed=(UInt32(26757677)*seed+UInt32(1))%UInt32(2147483648)
 push!(new,Float32(seed)/Float32(2147483648))
end
mismatch=findall(reinterpret(UInt32,old).!=reinterpret(UInt32,new))
@assert length(mismatch)==146
file=joinpath(@__DIR__,"historical_initializer_cpu_witness_f32.bin")
open(file,"w")do io;write(io,old);write(io,new);end
j=first(mismatch)
@printf("CPU_SOURCE_WITNESS {\"n\":257,\"mismatches\":%d,\"first_index0\":%d,\"historical\":%.17g,\"correct_source_contract\":%.17g,\"original_source_sha256\":\"%s\",\"executed_function_sha256\":\"%s\",\"witness_sha256\":\"%s\",\"GPU_launches\":0}\n",length(mismatch),j-1,Float64(old[j]),Float64(new[j]),bytes2hex(sha256(source)),bytes2hex(sha256(body)),bytes2hex(sha256(read(file))))
