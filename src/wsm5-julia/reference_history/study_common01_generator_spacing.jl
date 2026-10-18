using CUDA,StaticArrays
const STUDY_ROOT=@__DIR__
function study_main(alloc,launch)
 cases=open(joinpath(STUDY_ROOT,"fixtures_f32.bin"),"r")do io
  read(io,UInt32)==3||error("case count")
  [begin
   nx=read(io,UInt32);ny=read(io,UInt32);dt=read(io,Float32)
   a=[read!(io,Vector{Float32}(undef,Int(nx*ny*(f<10 ? 3 : 1))))for f in 0:14]
   raw=[alloc(fill(-777f0,length(x)+128))for x in a];d=[view(raw[f],65:64+length(a[f]))for f in 1:15]
   (nx=nx,ny=ny,dt=dt,a=a,raw=raw,d=d,bounds=Int32[1,nx,1,ny,1,3,1,nx,1,ny,1,3,1,nx,1,ny,1,3])
  end for ci in 1:3]
 end
 function reset(c)
  for f in 1:15;v=fill(-777f0,length(c.a[f])+128);v[65:64+length(c.a[f])]=c.a[f];copyto!(c.raw[f],v);end
  CUDA.device_synchronize()
 end
 for c in cases;reset(c);launch(c);CUDA.device_synchronize();end
 for trial in 1:5
  path="trial$(trial).bin";review="trial$(trial)_physics.json";seconds=0.
  open(path,"w")do io
   write(io,Float64(3))
   for c in cases
    reset(c);write(io,Float64[c.nx,c.ny,c.dt])
    for x in c.d;write(io,Float64.(Array(x)));end
   end
   for c in cases
    CUDA.device_synchronize();t=time_ns();launch(c);CUDA.device_synchronize();seconds+=(time_ns()-t)*1e-9
    for f in 1:15
     v=Array(c.raw[f]);n=length(c.a[f]);write(io,Float64.(v[65:64+n]));write(io,Float64.(v[1:64]));write(io,Float64.(v[65+n:end]))
    end
   end
  end
  run(`python3 $(joinpath(STUDY_ROOT,"study_check.py")) $path $review`)
  metrics=open(io->read!(io,Vector{Float64}(undef,3)),review*".metrics")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",seconds,",\"correct\":true,\"checked_values\":212480,\"max_abs_error\":",metrics[1],",\"max_rel_error\":",metrics[2],",\"timing_region\":\"sum3_source_kernel_context_completion\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":212480,\"byte_order\":\"little\"},\"physics_review\":\"",review,"\"}")
 end
end
