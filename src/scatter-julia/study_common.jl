@inline function applyop(old,v,::Val{OP}) where OP
 OP==1&&return old*v
 OP==2&&return (old isa Integer ? div(old,v) : old/v)
 OP==3&&return old+v
 OP==4&&return min(old,v)
 return max(old,v)
end
initial(::Type{T},op) where T = op<=2 ? one(T) : op==3 ? zero(T) : op==4 ? (T<:AbstractFloat ? floatmax(T) : typemax(T)) : (T<:AbstractFloat ? -floatmax(T) : typemin(T))
function study(run!)
 n=4096;outn=n÷2;repeat=10
 indices=Int64[mod(i*17+3,outn) for i in 0:n-1];di=CuArray(indices)
 cases=[(T,op) for T in (Int32,Int64,Float32,Float64) for op in 1:5]
 buffers=Any[];inputs=Any[];initials=Any[];expected=Any[]
 for (T,op) in cases
  U=sizeof(T)==4 ? UInt32 : UInt64;src=T[mod(i,3)==0 ? -1 : 1 for i in 0:n-1];init=fill(initial(T,op),outn);want=copy(init)
  for i in 1:n;j=indices[i]+1;want[j]=applyop(want[j],src[i],Val(op));end
  push!(buffers,CuArray(reinterpret(U,init)));push!(inputs,CuArray(src));push!(initials,collect(reinterpret(U,init)));push!(expected,want)
  run!(di,inputs[end],buffers[end],Int32(n),Val(T),Val(op))
 end
 CUDA.synchronize()
 for trial in 1:5
  times=Float64[];values=Float64[];correct=true
  for (c,(T,op)) in enumerate(cases)
   data=buffers[c];total=0.
   for _ in 1:repeat
    copyto!(data,initials[c]);CUDA.synchronize();start=time_ns();run!(di,inputs[c],data,Int32(n),Val(T),Val(op));CUDA.synchronize();total+=(time_ns()-start)*1e-9
   end
   push!(times,total/repeat);got=collect(reinterpret(T,Array(data)));correct &=got==expected[c]&&all(isfinite,got);append!(values,Float64.(got))
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,values),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",sum(times),",\"correct\":",correct,",\"checked_values\":",length(values),",\"max_abs_error\":",correct ? 0 : 1,",\"timing_region\":\"sum_twenty_individually_synchronized_kernel_means_excluding_reset\",\"jit_excluded\":true,\"internal_repeats\":",repeat,",\"subkernel_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(values),",\"byte_order\":\"little\"}}")
  correct||error("Exact scatter CPU oracle failed")
 end
end
