const SENT=-8192.25
function study(allocate,launch_stage,launch_reduce)
 fixture=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixtures_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));states=[];off=1
 function buffer(v,s)
  h=vcat(fill(s,4),v,fill(s,4));a=allocate(eltype(v),length(h));copyto!(a,h);return(a=a,v=view(a,5:length(a)-4),h=h,s=s)
 end
 while off<=length(fixture)
  n,it=Int.(fixture[off:off+1]);off+=3;N=n^3;nb=N÷64;speed=fixture[off:off+N-1];off+=N;init=fixture[off:off+N-1];off+=N;vol=falses(nb);vol[1]=true;list=fill(UInt32(0xdeadbeef),nb);list[1]=0
  push!(states,(n=n,it=it,speed=buffer(speed,SENT),mask=buffer(fill(true,N),false),a=buffer(init,SENT),b=buffer(init,SENT),con=buffer(fill(true,N),false),vol=buffer(collect(vol),false),list=buffer(list,UInt32(0xdeadbeef))))
 end
 function solve(c)
  n=c.n;nb=n^3÷64;nx=n÷4;list=fill(UInt32(0xdeadbeef),nb);listed=falses(nb);vol=falses(nb);list[1]=0;listed[1]=vol[1]=true;active=1;iter=0;processed=0
  while active>0
   iter+=1;iter<=1000||error("IterationLimit");processed+=active;copyto!(c.list.v,1,list,1,active)
   launch_stage(false,c.speed.v,c.mask.v,c.a.v,c.b.v,c.con.v,c.list.v,n,c.it,active,active)
   launch_reduce(c.con.v,c.vol.v,c.list.v,active)
   copyto!(vol,c.vol.v);old=active
   for j in 1:old
    cur=Int(list[j])
    if !vol[cur+1]
     neighbors=(cur<nx*nx ? cur : cur-nx*nx,cur+nx*nx>=nb ? cur : cur+nx*nx,cur<nx ? cur : cur-nx,cur+nx>=nb ? cur : cur+nx,cur%nx==0 ? cur : cur-1,(cur+1)%nx==0 ? cur : cur+1)
     for v in neighbors
      if !listed[v+1];listed[v+1]=true;active+=1;list[active]=UInt32(v);end
     end
    end
   end
   copyto!(c.list.v,1,list,1,active)
   launch_stage(true,c.speed.v,c.mask.v,c.b.v,c.a.v,c.con.v,c.list.v,n,1,old,active)
   launch_reduce(c.con.v,c.vol.v,c.list.v,active)
   copyto!(vol,c.vol.v);active=0
   for j in 1:nb
    listed[j]=vol[j]
    if vol[j];active+=1;list[active]=UInt32(j-1);end
   end
  end
  CUDA.synchronize(CUDA.context());return iter,processed
 end
 for c in states;solve(c);end
 for trial in 1:5
  out=copy(fixture);elapsed=0.;ok=true;err=0.;oi=1
  for c in states
   for buf in(c.a,c.b,c.con,c.vol,c.list);copyto!(buf.a,buf.h);end
   CUDA.synchronize(CUDA.context());t=time_ns();iter,processed=solve(c);elapsed+=(time_ns()-t)*1e-9
   sol=Array(c.a.v);expected=oracle[oi:oi+length(sol)-1];oi+=length(sol);es=abs.(sol.-expected);err=max(err,maximum(es));ok&=all(isfinite,sol)&&all(x>=0 for x in sol)&&sol[1]==0&&all(es.<=1e-5.+1e-5.*abs.(expected))
   for (j,buf) in enumerate((c.speed,c.mask,c.a,c.b,c.con,c.vol,c.list))
    h=Array(buf.a);ok&=all(x==buf.s for x in h[1:4])&&all(x==buf.s for x in h[end-3:end]);if j<=2;ok&=h==buf.h;end;append!(out,Float64.(h))
   end
   push!(out,Float64(iter));push!(out,Float64(processed))
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum3_full_active_list_FIM_solver_host_scheduling_copies_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("FullFIMIndependentGodunovAndGuards")
 end
end
