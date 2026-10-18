function fixture(caseid)
 points=Cmplx64[];bases=(-1,0,1,15,32,62,63,64,47)
 for i in 0:63
  push!(points,Cmplx64(bases[i%9+1]+(i%8)/8,bases[(i*5)%9+1]+(i÷8)/8))
 end
 # Actual native qsort key swaps sub-bin axes; retain source locality ordering.
 sort!(points,by=v->Int(floor(8*(v.y-floor(v.y))))+8Int(floor(8*(v.x-floor(v.x)))))
 image=fill(Cmplx64(0.,0.),8256)
 for y in 0:63,x in 0:63
  image[2080+y*64+x+1]=Cmplx64(((x*3+y*5)%31-15)/16,((x*7+y*2)%29-14)/16)
 end
 gcf=Vector{Cmplx64}(undef,65536)
 for sy in 0:7,sx in 0:7,y in 0:31,x in 0:31
  gcf[(sy*8+sx)*1024+y*32+x+1]=caseid==2 ? Cmplx64(x==16&&y==16 ? 1. : 0.,0.) : Cmplx64(((x*3+y*5+sx*7+sy*11)%31-15)/64,((x*7+y*3+sx*5+sy*2)%29-14)/64)
 end
 return points,image,gcf
end
function exact_oracle(points,image,gcf)
 out=Float64[]
 for inn in points
  mx=Int(floor(inn.x));my=Int(floor(inn.y));sx=Int(floor(8*(inn.x-mx)));sy=Int(floor(8*(inn.y-my)));realnumerator=Int64(0);imagnumerator=Int64(0)
  for dy in -16:15,dx in -16:15
   x=mx+dx;y=my+dy
   if !(0<=x<64&&0<=y<64);continue;end
   v=image[2080+y*64+x+1];c=gcf[(sy*8+sx)*1024+(dy+16)*32+dx+16+1];vr=round(Int64,v.x*16);vi=round(Int64,v.y*16);cr=round(Int64,c.x*64);ci=round(Int64,c.y*64);realnumerator+=vr*cr-vi*ci;imagnumerator+=vr*ci+vi*cr
  end
  push!(out,realnumerator/1024,imagnumerator/1024)
 end
 return out
end
function encode!(out,values)
 for v in values;push!(out,v.x,v.y);end
 return out
end
function study(launch,allocate)
 cases=[];prefix=Float64[]
 for id in 1:2
  points,image,gcf=fixture(id);dp=allocate(Cmplx64,64);di=allocate(Cmplx64,8256);dg=allocate(Cmplx64,65536);out=allocate(Cmplx64,64);copyto!(dp,points);copyto!(di,image);copyto!(dg,gcf);for x in (points,image,gcf);encode!(prefix,x);end;append!(prefix,zeros(Float64,128));append!(prefix,[64,64,32,8,2080,3,id]);push!(cases,(dp,di,dg,out,exact_oracle(points,image,gcf)))
 end
 function case!(c)
  dp,di,dg,out,want=c;CUDA.fill!(out,Cmplx64(0.,0.));CUDA.synchronize(CUDA.context());t=time_ns();for _ in 1:3;launch(out,dp,di,dg);end;CUDA.synchronize(CUDA.context());elapsed=(time_ns()-t)*1e-9/3;got=Float64[];encode!(got,Array(out));err=maximum(abs.(got.-want));return got,elapsed,err<=1e-7,err
 end
 for c in cases;got,t,ok,e=case!(c);ok||error("WarmupDegridMismatch");end
 for trial in 1:5
  out=copy(prefix);elapsed=0.;ok=true;err=0.
  for c in cases;got,t,pass,e=case!(c);append!(out,got);elapsed+=t;ok&=pass;err=max(err,e);end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum_two_three_kernel_means\",\"jit_excluded\":true,\"internal_repeats\":3,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("FullComplexDegridExactRationalMismatch")
 end
end
