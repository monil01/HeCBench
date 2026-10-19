using CUDA,Libdl
function coefficients(v,n)
 x=0f0;y=0f0;xy=0f0;xx=0f0
 for j in 1:4:length(v);x+=v[j];y+=v[j+1];xy+=v[j+2];xx+=v[j+3];end
 nf=Float32(n);denom=Float64(nf*xx-x*x)
 a=Float32(Float64(y*xx-x*xy)/denom);b=Float32(Float64(nf*xy-x*y)/denom)
 return y/nf,a,b
end
function study(alloc,first,second)
 cs=[]
 for n in (63,126,96453)
  d=Float32[]
  for i in 0:n-1
   x=Float32(mod(i,63)-31)/16f0;noise=n==63 ? 0f0 : Float32(mod(i*17,7)-3)/64f0
   push!(d,x);push!(d,1.25f0+.75f0*x+noise)
  end
  rd=alloc(fill(-777f0,2n+128));rq=alloc(fill(-777f0,4div(n,63)+128));rz=alloc(fill(-777f0,2div(n,63)+128))
  push!(cs,(n=n,repeat=n==96453 ? 7 : 3,host=d,rd=rd,rq=rq,rz=rz,d=view(rd,65:64+2n),q=view(rq,65:64+4div(n,63)),z=view(rz,65:64+2div(n,63))))
 end
 reset(c)=(fill!(c.rd,-777f0);copyto!(c.d,c.host);fill!(c.rq,-777f0);fill!(c.rz,-777f0))
 for c in cs;reset(c);first(c);second(c,0f0,1f0,.75f0);CUDA.device_synchronize();end
 check=Libdl.dlsym(Libdl.dlopen(joinpath(pwd(),"studycheck.so")),:study_check)
 for trial in 1:5
  path="trial$(trial).bin";seconds=0.
  open(path,"w")do io
   write(io,3f0)
   for c in cs;reset(c);CUDA.device_synchronize();write(io,Float32[c.n,c.repeat]);write(io,Array(c.d));end
   for c in cs
    CUDA.device_synchronize();start=time_ns()
    for k in 1:c.repeat;first(c);end
    CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9
    q=Array(c.q);m,a,b=coefficients(q,c.n)
    CUDA.device_synchronize();start=time_ns()
    for k in 1:c.repeat;second(c,m,a,b);end
    CUDA.device_synchronize();seconds+=(time_ns()-start)*1e-9
    z=Array(c.z);write(io,q);write(io,z);vx=0f0;vy=0f0
    for j in 1:2:length(z);vx+=z[j];vy+=z[j+1];end
    write(io,Float32[m,a,b,trunc(Int,vy/vx*100f0)])
    reinterpret(UInt32,Array(c.d))==reinterpret(UInt32,c.host)||error("readonlyinput")
    for raw in (c.rd,c.rq,c.rz);h=Array(raw);write(io,h[1:64]);write(io,h[end-63:end]);end
   end
  end
  met=zeros(Float64,3);ccall(check,Cint,(Cstring,Ptr{Float64}),path,met)==0||error("independent/fullstatecheck")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":true,\"checked_values\":203659,\"max_abs_error\":",met[1],",\"max_rel_error\":",met[2],",\"independent_OLS_max_abs\":",met[3],",\"timing_region\":\"sum3_two_stage_kernel_loop_context_completion\",\"jit_excluded\":true,\"internal_repeats\":26,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float32\",\"count\":203659,\"byte_order\":\"little\"}}")
 end
end
