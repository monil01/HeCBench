include(joinpath(@__DIR__,"study_mpi.jl"))
const SENT=-8192.25
function launch_for(c,launch,which,args)
 nx,nz=c.nx,c.nz
 if which in (:fx,);blocks=(cld(Int(nx)+1,16),cld(Int(nz),16));threads=(16,16)
 elseif which in (:fz,);blocks=(cld(Int(nx),16),cld(Int(nz)+1,16));threads=(16,16)
 elseif which in (:pack,:unpack);blocks=(1,cld(Int(nz),16),4);threads=(16,16,1)
 elseif which==:zhalo;blocks=(cld(Int(nx)+4,16),1);threads=(16,16)
 elseif which==:mass;blocks=(cld(Int(nx),16),cld(Int(nz),16));threads=(16,16)
 else;blocks=(cld(Int(nx),16),cld(Int(nz),16),4);threads=(16,16,1)
 end
 launch(which,args,blocks,threads)
end
function halo_mpi!(c,forcing,launch)
 GC.@preserve c begin
  mpi_irecv!(c.host[3],0,c.reqrecv,1);mpi_irecv!(c.host[4],1,c.reqrecv,2)
  launch_for(c,launch,:pack,(forcing,c.A[5],c.A[6],c.nx,c.nz,HS))
  copyto!(c.host[1],c.A[5]);copyto!(c.host[2],c.A[6])
  mpi_isend!(c.host[1],1,c.reqsend,1);mpi_isend!(c.host[2],0,c.reqsend,2);mpi_wait!(c.reqrecv)
  copyto!(c.A[7],c.host[3]);copyto!(c.A[8],c.host[4])
  launch_for(c,launch,:unpack,(forcing,c.A[7],c.A[8],c.nx,c.nz,HS));mpi_wait!(c.reqsend)
 end
 return
end
function substep!(c,forcing,out,subdt,axis,fulldt,launch)
 if axis==1
  halo_mpi!(c,forcing,launch);hv=-HV_BETA*c.dx/(16*fulldt)
  launch_for(c,launch,:fx,(forcing,c.A[3],c.H[1],c.H[2],hv,c.nx,c.nz,HS))
  launch_for(c,launch,:tx,(c.A[3],c.A[4],c.nx,c.nz,Int32(c.dx)))
 else
  launch_for(c,launch,:zhalo,(forcing,c.nx,c.nz,HS));hv=-HV_BETA*c.dz/(16*fulldt)
  launch_for(c,launch,:fz,(forcing,c.A[3],c.H[3],c.H[5],c.H[4],hv,c.nx,c.nz,HS))
  launch_for(c,launch,:tz,(forcing,c.A[3],c.A[4],c.nx,c.nz,HS,Int32(c.dz)))
 end
 launch_for(c,launch,:update,(c.A[1],out,c.A[4],c.nx,c.nz,HS,subdt))
end
function integrate!(c,launch)
 for (step,dt) in enumerate(c.schedule)
  for axis in (isodd(step) ? (1,2) : (2,1))
   substep!(c,c.A[1],c.A[2],dt/3,axis,dt,launch)
   substep!(c,c.A[2],c.A[2],dt/2,axis,dt,launch)
   substep!(c,c.A[2],c.A[1],dt,axis,dt,launch)
  end
 end
 CUDA.synchronize(CUDA.context())
end
function diagnostics(c,launch)
 fill!(c.mass[2],0.);fill!(c.energy[2],0.)
 launch_for(c,launch,:mass,(c.mass[2],c.energy[2],c.A[1],c.H[1],c.H[2],c.nx,c.nz,c.dx,c.dz))
 localdata=[Array(c.mass[2])[1],Array(c.energy[2])[1]]
 return mpi_sum(localdata)
end
function reset!(c)
 for i in 1:8;copyto!(c.A[i],c.initial_mutable[i]);end
 for h in c.host;fill!(h,0.);end
 CUDA.synchronize(CUDA.context())
end
function study(allocate,launch)
 mpi_init()
 CUDA.stream!(CUDA.legacy_stream()) do
 f=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixtures_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));nc=Int(f[1]);pos=2;cases=[]
 function guarded(data)
  raw=allocate(Float64,length(data)+8);fill!(raw,SENT);copyto!(view(raw,5:length(raw)-4),data);(raw,view(raw,5:length(raw)-4))
 end
 for ci in 1:nc
  nx,nz,steps=Int.(f[pos:pos+2]);dx,dz,full=f[pos+3:pos+5];pos+=6;schedule=f[pos:pos+steps-1];pos+=steps;ns=4*(nx+4)*(nz+4);initial=f[pos:pos+ns-1];pos+=ns;hy=[]
  for n in [nz+4,nz+4,nz+1,nz+1,nz+1];push!(hy,copy(f[pos:pos+n-1]));pos+=n;end
  initial_mutable=[initial,copy(initial),zeros(4*(nx+1)*(nz+1)),zeros(4*nx*nz),zeros(8*nz),zeros(8*nz),zeros(8*nz),zeros(8*nz)]
  pairs=guarded.(initial_mutable);hpairs=guarded.(hy);c=(nx=Int32(nx),nz=Int32(nz),steps=steps,dx=dx,dz=dz,schedule=schedule,initial=initial,hy=hy,initial_mutable=initial_mutable,raw=[x[1] for x in pairs],A=[x[2] for x in pairs],hraw=[x[1] for x in hpairs],H=[x[2] for x in hpairs],host=[zeros(8*nz) for _ in 1:4],reqrecv=fill(Ptr{Cvoid}(0),2),reqsend=fill(Ptr{Cvoid}(0),2),mass=guarded([0.]),energy=guarded([0.]))
  push!(cases,c)
 end
 # Compilewarmup all9kernels, everygeometry andfullMPIpath before anytrial.
 for c in cases;reset!(c);diagnostics(c,launch);integrate!(c,launch);diagnostics(c,launch);end
 for trial in 1:5
  payload=copy(f);ok=true;maxerror=0.;total=0.;times=Float64[];oi=1
  for c in cases
   reset!(c);initial_diag=diagnostics(c,launch);CUDA.synchronize(CUDA.context());start=time_ns();integrate!(c,launch);tm=(time_ns()-start)*1e-9;total+=tm;push!(times,tm);final_diag=diagnostics(c,launch)
   for (raw,original) in zip(c.hraw,c.hy)
    got=Array(raw);ok&=got[5:end-4]==original&&all(==(SENT),got[1:4])&&all(==(SENT),got[end-3:end]);append!(payload,got)
   end
   values=[]
   for raw in c.raw
    got=Array(raw);ok&=all(==(SENT),got[1:4])&&all(==(SENT),got[end-3:end]);append!(payload,got);push!(values,got[5:end-4])
   end
   diag=vcat(initial_diag,final_diag);append!(payload,diag);push!(values,diag)
   for got in values
    expected=oracle[oi:oi+length(got)-1];oi+=length(got)
    for (x,y) in zip(got,expected);e=abs(x-y);maxerror=max(maxerror,e);ok&=isfinite(x)&&e<=2e-8+2e-11*abs(y);end
   end
   ok&=abs(final_diag[1]-initial_diag[1])<=1e-12*initial_diag[1]
   for pair in [c.mass,c.energy]
    raw=Array(pair[1]);ok&=all(==(SENT),raw[1:4])&&all(==(SENT),raw[end-3:end]);append!(payload,raw)
   end
   for i in 1:4;ok&=c.host[i]==values[i+4];append!(payload,c.host[i]);end
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,payload),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",total,",\"correct\":",ok,",\"checked_values\":",length(payload),",\"max_abs_error\":",maxerror,",\"timing_region\":\"sum3_full_splitRK_MPIself_halo_pipeline_complete\",\"jit_excluded\":true,\"internal_repeats\":1,\"timed_kernel_calls\":162,\"case_times_seconds\":",times,",\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(payload),",\"byte_order\":\"little\"}}")
  ok||error("FullFiniteVolumeRKStatesFluxTendMPIBuffersIndependentOracleMassGuards")
 end
 end
 mpi_finalize()
end
