function lcg_next(seed::UInt64)
    a = UInt64(2806196910506780709)
    c = UInt64(1)
    # Modulo 2^63 is equivalent to clearing the high bit for UInt64 arithmetic.
    return (a * seed + c) & UInt64(0x7fffffffffffffff)
end


function selected(seed,k,cols)
 mask=zeros(UInt8,cols);count=0
 while count<k
  seed=lcg_next(seed);idx=Int32(floor(Float64(seed)/9223372036854775808.0*cols));idx=min(idx,Int32(cols-1))
  if mask[idx+1]==0;mask[idx+1]=1;count+=1;end
 end
 mask
end
native_policy=selected(UInt64(1234),10,2000);old_block1=selected(UInt64(1234)+UInt64(1),10,2000)
write(joinpath(@__DIR__,"historical_seed_cpu_witness_u8.bin"),vcat(native_policy,old_block1))
println("HISTORICAL_CPU_SEED_WITNESS mismatches=",count(!iszero,native_policy .!=old_block1)," GPU_launches=0 native_indices=",findall(!iszero,native_policy).-1," historical_block1_indices=",findall(!iszero,old_block1).-1)
