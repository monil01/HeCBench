include(joinpath(@__DIR__,"study_primitives.jl"))
const STUDY_CASES=((1,7),(257,7),(16387,7),(1027,100))
const STUDY_REPEATS=10
function real_directions()
 output=UInt32[]
 for (dim,(degree,a,m)) in enumerate(STUDY_PRIMITIVES)
  v=zeros(UInt32,32)
  if dim==1
   for i in 0:31;v[i+1]=UInt32(1)<<(31-i);end
  else
   for i in 0:degree-1;v[i+1]=m[i+1]<<(31-i);end
   for i in degree:31
    value=v[i-degree+1]⊻(v[i-degree+1]>>degree)
    for j in 1:degree-1;value⊻=((a>>(degree-1-j))&UInt32(1))*v[i-j+1];end
    v[i+1]=value
   end
  end
  append!(output,v)
 end
 output
end
function geometry(n,d)
 target=d<96 ? 96 : 1
 target>n÷128&&(target=cld(n,128));gx=1
 while gx<target;gx*=2;end
 gx
end
function expected(n,d,dirs)
 output=Float32[]
 for dim in 0:d-1,i in 0:n-1
  gray=UInt32(i)⊻(UInt32(i)>>1);value=UInt32(0)
  for bit in 0:31;(gray&(UInt32(1)<<bit))!=0&&(value⊻=dirs[dim*32+bit+1]);end
  push!(output,Float32(value)*2.3283064f-10)
 end
 reinterpret(UInt32,output)
end
function study(run!)
 dirs=real_directions();device_dirs=CuArray(dirs);cases=[(n,d,geometry(n,d),CUDA.zeros(Float32,n*d),expected(n,d,dirs)) for (n,d) in STUDY_CASES]
 template=CUDA.zeros(UInt32,32)
 for(n,d,gx,out,want)in cases;run!(device_dirs,out,Int32(n),Int32(d),Int32(gx),template);CUDA.synchronize();end
 for trial in 1:5
  values=copy(dirs);elapsed=0.;ok=true
  for(n,d,gx,out,want)in cases
   CUDA.synchronize();t=time_ns()
   for _ in 1:STUDY_REPEATS;run!(device_dirs,out,Int32(n),Int32(d),Int32(gx),template);end
   CUDA.synchronize();elapsed+=(time_ns()-t)*1e-9/STUDY_REPEATS
   got=copy(reinterpret(UInt32,Array(out)));ok&=got==want;append!(values,got)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,values),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":222415,\"max_abs_error\":0,\"timing_region\":\"sum_four_sobol_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"uint32\",\"count\":222415,\"byte_order\":\"little\"}}")
  ok||Base.error("SobolGraycodedirectoraclefailed")
 end
end
