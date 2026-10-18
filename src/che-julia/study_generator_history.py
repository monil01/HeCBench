from pathlib import Path
import math,struct,json,shutil,hashlib
R=Path('/home/imo/HeCBench_julia');p=R/'src/che-julia';d=p/'che-julia-cuda.jl';d.mkdir(exist_ok=True);shutil.copy2(p/'main.jl',d/'original_main.jl');shutil.copy2(p/'Makefile',p/'original_Makefile');shutil.copy2(R/'src/che-cuda/LICENSE',p/'LICENSE');(p/'main.jl').write_text('include(joinpath(@__DIR__,"che-julia-cuda.jl","main.jl"))\n')
native=(R/'src/che-cuda/kernels.h').read_text();span=native[native.index('__device__ double Laplacian'):];(p/'native_kernels.inc').write_text('// Verbatim device helper/kernel span from che-cuda/kernels.h; dimensions supplied by includer.\n'+span)
cases=[(16,8,8,1,0),(19,9,7,3,0),(16,8,8,3,1),(16,8,8,3,2)];fixture=[];oracle=[];mass=[];energies=[]
for nx,ny,nz,steps,mode in cases:
 v=nx*ny*nz;c=[]
 for z in range(nz):
  for y in range(ny):
   for x in range(nx):
    c.append(((x*17+y*11+z*7)%31-15)/64 if mode==0 else .25 if mode==1 else math.cos(2*math.pi*x/nx)/16)
 fixture.extend(c);initialmass=math.fsum(c)
 def at(x,y,z):return (z%nz*ny+y%ny)*nx+x%nx
 def lap(a,x,y,z):return math.fsum(a[at(x+dx,y+dy,z+dz)]-a[at(x,y,z)] for dx,dy,dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])
 for st in range(steps):
  mu=[];ff=[]
  for z in range(nz):
   for y in range(ny):
    for x in range(nx):
     i=at(x,y,z);q=c[i];mu.append(q*q*q-q-.5*lap(c,x,y,z));grads=[(c[at(x+dx,y+dy,z+dz)]-c[at(x-dx,y-dy,z-dz)])/2 for dx,dy,dz in [(1,0,0),(0,1,0),(0,0,1)]];ff.append(.25*q**4-.5*q*q+.25*sum(g*g for g in grads))
  cn=[c[at(x,y,z)]+.01*lap(mu,x,y,z) for z in range(nz)for y in range(ny)for x in range(nx)];old=c;c=cn
 oracle.extend(c+old+mu+ff);mass.append({'initial':initialmass,'final':math.fsum(c),'absolute_drift':abs(math.fsum(c)-initialmass)});energies.append(math.fsum(ff))
for name,data in [('fixtures_f64.bin',fixture),('oracle_f64.bin',oracle)]:
 with(p/name).open('wb')as io:io.write(struct.pack('<'+'d'*len(data),*data))
(p/'independent_oracle.json').write_text(json.dumps({'method':'Independent simplified free-energy derivative mu=c^3-c-(1/2)Laplacian(c), f=(1/4)c^4-(1/2)c^2+(1/4)|centered-gradient(c)|²; periodic neighbor-difference fsum, explicit Euler. Not a literal translation of source coefficient expression.','source_coefficient_identity':'eAA=eBB=-2/9,eAB=2/9 reduce local potential to (1/4)c^4-(1/2)c² mathematically; actual rounded source coefficients compared at declared1e-11 bound','cases':cases,'mass_conservation':mass,'max_mass_drift':max(x['absolute_drift']for x in mass),'constant_state_exact_expected':{'c':.25,'mu':-.234375,'f':-.0302734375},'scientific_scope':'Discrete periodic conserved-phase operator and short-step full fields, not256cubed coarsening or physical-material validation'},indent=2)+'\n')
mathjl=r'''@inline at3(x,y,z,nx,ny)=(Int(z)*Int(ny)+Int(y))*Int(nx)+Int(x)+1
@inline function lap(a,x,y,z,nx,ny,nz)
 xp=x+1==nx ? Int32(0) : x+Int32(1);xn=x==0 ? nx-Int32(1) : x-Int32(1)
 yp=y+1==ny ? Int32(0) : y+Int32(1);yn=y==0 ? ny-Int32(1) : y-Int32(1)
 zp=z+1==nz ? Int32(0) : z+Int32(1);zn=z==0 ? nz-Int32(1) : z-Int32(1)
 @inbounds center=a[at3(x,y,z,nx,ny)]
 @inbounds xx=a[at3(xp,y,z,nx,ny)]+a[at3(xn,y,z,nx,ny)]-2.0*center
 @inbounds yy=a[at3(x,yp,z,nx,ny)]+a[at3(x,yn,z,nx,ny)]-2.0*center
 @inbounds zz=a[at3(x,y,zp,nx,ny)]+a[at3(x,y,zn,nx,ny)]-2.0*center
 return xx+yy+zz
end
@inline function point!(x,y,z,stage,cold,cnew,mu,ff,nx,ny,nz)
 if x<nx&&y<ny&&z<nz
  i=at3(x,y,z,nx,ny);ea=-2.0/9.0;eb=-2.0/9.0;eab=2.0/9.0;gamma=.5
  @inbounds if stage==1
   c=cold[i];mu[i]=4.5*((c+1.0)*ea+(c-1)*eb-2.0*c*eab)+3.0*c+c*c*c-gamma*lap(cold,x,y,z,nx,ny,nz)
  elseif stage==2
   c=cold[i];free=(9.0/4.0)*((c*c+2.0*c+1.0)*ea+(c*c-2.0*c+1.0)*eb+2.0*(1.0-c*c)*eab)+(3.0/2.0)*c*c+(3.0/12.0)*c*c*c*c
   xp=x+1==nx ? Int32(0) : x+Int32(1);xn=x==0 ? nx-Int32(1) : x-Int32(1);yp=y+1==ny ? Int32(0) : y+Int32(1);yn=y==0 ? ny-Int32(1) : y-Int32(1);zp=z+1==nz ? Int32(0) : z+Int32(1);zn=z==0 ? nz-Int32(1) : z-Int32(1)
   gx=(cold[at3(xp,y,z,nx,ny)]-cold[at3(xn,y,z,nx,ny)])/2.0;gy=(cold[at3(x,yp,z,nx,ny)]-cold[at3(x,yn,z,nx,ny)])/2.0;gz=(cold[at3(x,y,zp,nx,ny)]-cold[at3(x,y,zn,nx,ny)])/2.0
   ff[i]=free+(gamma/2.0)*(gx*gx+gy*gy+gz*gz)
  elseif stage==3
   cnew[i]=cold[i]+.01*1.0*lap(mu,x,y,z,nx,ny,nz)
  else
   tmp=cnew[i];cnew[i]=cold[i];cold[i]=tmp
  end
 end
 return
end
''';(p/'study_math.jl').write_text(mathjl)
common=r'''const CASES=((16,8,8,1,0),(19,9,7,3,0),(16,8,8,3,1),(16,8,8,3,2));const SENT=-8192.25
function study(allocate,launch)
 fixture=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixtures_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));states=[];off=0
 for(nx,ny,nz,steps,mode)in CASES
  v=nx*ny*nz;raw=[allocate(Float64,v+2) for k in 1:4];a=[view(q,2:v+1) for q in raw];input=copy(fixture[off+1:off+v]);push!(states,(nx,ny,nz,steps,mode,raw,a,input));off+=v
 end
 function run(c)
  nx,ny,nz,steps,mode,raw,a,input=c;v=nx*ny*nz
  for k in 1:4;h=fill(SENT,v+2);k==1&&(h[2:end-1]=input);copyto!(raw[k],h);end
  CUDA.synchronize(CUDA.context());start=time_ns()
  for step in 1:steps,stage in 1:4;launch(stage,a...,Int32(nx),Int32(ny),Int32(nz));end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9;arrays=Array.(raw);guards=Float64[];ok=true
  for h in arrays;append!(guards,Float64[h[1],h[end]]);ok&=h[1]===SENT&&h[end]===SENT;end
  vals=vcat([h[2:end-1] for h in arrays]...);masserr=abs(sum(Float64,arrays[1][2:end-1])-sum(Float64,input));ok&=masserr<=1e-10
  return vals,guards,elapsed,ok,masserr
 end
 for c in states;run(c);end
 for trial in 1:5
  out=copy(fixture);oi=0;elapsed=0.;ok=true;err=0.;masserr=0.
  for c in states
   vals,guards,t,pass,me=run(c);ref=oracle[oi+1:oi+length(vals)];e=maximum(abs.(vals.-ref));ok&=pass&&all(isfinite,vals)&&all(abs.(vals.-ref).<=1e-11 .+1e-11.*abs.(ref));err=max(err,e);masserr=max(masserr,me);oi+=length(vals);append!(out,vals);append!(out,guards);elapsed+=t
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_MASS_DRIFT ",masserr)
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum4_preallocated_periodic_fourlaunch_step_pipelines\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("CheDiscreteFieldsMassOrExactGuards")
 end
end
''';(p/'study_common.jl').write_text(common)
(d/'main.jl').write_text('using CUDA\ninclude(joinpath(@__DIR__,"..","study_math.jl"))\nfunction che_kernel!(stage,args...)\n x=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x-1);y=Int32((blockIdx().y-1)*blockDim().y+threadIdx().y-1);z=Int32((blockIdx().z-1)*blockDim().z+threadIdx().z-1)\n point!(x,y,z,stage,args...)\n return\nend\nfunction launch(stage,cold,cnew,mu,ff,nx,ny,nz)\n @cuda threads=(16,4,4) blocks=(cld(nx,16),cld(ny,4),cld(nz,4)) che_kernel!(Val(stage),cold,cnew,mu,ff,nx,ny,nz)\nend\ninclude(joinpath(@__DIR__,"..","study_common.jl"))\nstudy((T,n)->CUDA.zeros(T,n),launch)\n')
# stage specialized via Val for four separate authored entry specializations, preserving source stage boundaries.
s=(p/'study_math.jl').read_text().replace('function point!(x,y,z,stage,','function point!(x,y,z,::Val{stage},').replace('ff,nx,ny,nz)\n if','ff,nx,ny,nz) where {stage}\n if');(p/'study_math.jl').write_text(s)
native=r'''#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstdio>
#include <string>
#include <algorithm>
'''
for name,dims in [('A',(16,8,8)),('B',(19,9,7))]:
 nx,ny,nz=dims;native+=f'#define DATAXSIZE {nx}\n#define DATAYSIZE {ny}\n#define DATAZSIZE {nz}\nnamespace {name}{{\n#include "native_kernels.inc"\n}}\n#undef DATAXSIZE\n#undef DATAYSIZE\n#undef DATAZSIZE\n'
 native+=f'void step{name}(std::vector<double*>&a){{using namespace {name};using P=double(*)[{ny}][{nx}];dim3 block(16,4,4),grid(({nx}+15)/16,({ny}+3)/4,({nz}+3)/4);chemicalPotential<<<grid,block>>>((P)a[0],(P)a[2],1.,1.,1.,.5,-2./9.,-2./9.,2./9.);localFreeEnergyFunctional<<<grid,block>>>((P)a[0],(P)a[3],1.,1.,1.,.5,-2./9.,-2./9.,2./9.);cahnHilliard<<<grid,block>>>((P)a[1],(P)a[0],(P)a[2],1.,.01,1.,1.,1.);Swap<<<grid,block>>>((P)a[1],(P)a[0]);}}\n'
native+=r'''void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
std::vector<double>load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);size_t n=f.tellg();f.seekg(0);std::vector<double>v(n/8);f.read((char*)v.data(),n);return v;}
struct C{int nx,ny,nz,steps,mode;};C cs[]={{16,8,8,1,0},{19,9,7,3,0},{16,8,8,3,1},{16,8,8,3,2}};struct S{std::vector<double*>raw,a;};
int main(){auto fixture=load(CHE_FIXTURE),oracle=load(CHE_ORACLE);std::vector<S>s(4);for(int c=0;c<4;c++)for(int k=0;k<4;k++){double*p;int v=cs[c].nx*cs[c].ny*cs[c].nz;ck(cudaMalloc(&p,(v+2)*8));s[c].raw.push_back(p);s[c].a.push_back(p+1);}
auto run=[&](int c){int v=cs[c].nx*cs[c].ny*cs[c].nz;size_t off=0;for(int j=0;j<c;j++)off+=cs[j].nx*cs[j].ny*cs[j].nz;std::vector<double>h(v+2,-8192.25);for(int k=0;k<4;k++){std::fill(h.begin()+1,h.end()-1,-8192.25);if(k==0)std::copy(fixture.begin()+off,fixture.begin()+off+v,h.begin()+1);ck(cudaMemcpy(s[c].raw[k],h.data(),(v+2)*8,cudaMemcpyHostToDevice));}ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int st=0;st<cs[c].steps;st++){if(cs[c].nx==16)stepA(s[c].a);else stepB(s[c].a);}ck(cudaGetLastError());ck(cudaDeviceSynchronize());return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};for(int c=0;c<4;c++)run(c);for(int trial=1;trial<=5;trial++){auto out=fixture;size_t oi=0,fi=0;double elapsed=0,err=0,maxmass=0;bool ok=true;for(int c=0;c<4;c++){elapsed+=run(c);int v=cs[c].nx*cs[c].ny*cs[c].nz;std::vector<double>guards;double mass=0,initialmass=0;for(int i=0;i<v;i++)initialmass+=fixture[fi+i];fi+=v;for(int k=0;k<4;k++){std::vector<double>h(v+2);ck(cudaMemcpy(h.data(),s[c].raw[k],(v+2)*8,cudaMemcpyDeviceToHost));ok&=h.front()==-8192.25&&h.back()==-8192.25;guards.push_back(h.front());guards.push_back(h.back());for(int i=0;i<v;i++){double e=std::abs(h[i+1]-oracle[oi]);err=std::max(err,e);ok&=std::isfinite(h[i+1])&&e<=1e-11+1e-11*std::abs(oracle[oi]);oi++;if(k==0)mass+=h[i+1];}out.insert(out.end(),h.begin()+1,h.end()-1);}maxmass=std::max(maxmass,std::abs(mass-initialmass));ok&=std::abs(mass-initialmass)<=1e-10;out.insert(out.end(),guards.begin(),guards.end());}std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)out.data(),out.size()*8);f.close();printf("STUDY_MASS_DRIFT %.17g\n",maxmass);printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"timing_region\":\"sum4_preallocated_periodic_fourlaunch_step_pipelines\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,elapsed,ok?"true":"false",out.size(),err,path.c_str(),out.size());if(!ok)return 3;}for(auto&t:s)for(auto*q:t.raw)ck(cudaFree(q));}
''';(p/'study_native.cu').write_text(native)
config={'benchmark':'che','configuration_id':'che_periodic_16x8x8_19x9x7_steps1_3_varied_constant_fourier_v1','gpu':6,'timeout':900,'inputs_shared_verified':True,'input_prefix_count':len(fixture),'timed_regions_equivalent_verified':True,'input_contract':{'cases':cases,'precision':'Float64','parameters':{'dx_dy_dz':1.,'dt':.01,'gamma':.5,'D':1.,'eAA_eBB':-2/9,'eAB':2/9},'fixture':'Sharedserializedsigneddyadic,constant0.25,singlexFouriermode amplitude1/16','outputs':'All4rawfields current/previousc,mu,localfreeenergy and32exactguards; sourceperiodicwrap allfaces','geometry':[16,4,4],'timing':'Sum4preallocatedsource4launchstep pipelines withcontextcompletion; excludesoriginal diagnosticD2H/integrals/fileI/O consistentlyall4','scope':'Namedsmallgridsand1/3steps, includingpartialtiles; no256cubed/25000stepCLI orcoarseningvalidation','mass_conservation_absolute_bound':1e-10},'atol':1e-11,'rtol':1e-11,'numerical_evidence_strength':'all4raw_fields_independent_simplified_discrete_freeenergy_derivative_and_periodic_mass','models':{}}
deps=[str(p/q)for q in ['fixtures_f64.bin','oracle_f64.bin']]
config['models']['cuda']={'structured':True,'cwd':str(p),'source_dependencies':deps+[str(p/'native_kernels.inc'),str(R/'src/che-cuda/kernels.h')],'build':['nvcc','-std=c++17','-O3','-arch={arch}','-DCHE_FIXTURE="'+str(p/'fixtures_f64.bin')+'"','-DCHE_ORACLE="'+str(p/'oracle_f64.bin')+'"','study_native.cu','-o','study_native'],'command':['./study_native'],'verification_method':'full_discrete_fields_and_independent_periodic_mass_freeenergy_identity','native_check_description':'Kernelbodyspanunchanged, only includercompiletime dimensions; independent algebraically reduced chemicalpotential/freeenergy and periodicmass'}
config['models']['cuda_jl']={'structured':True,'cwd':str(d),'source_dependencies':deps+[str(p/'study_math.jl'),str(p/'study_common.jl')],'command':['julia','--project='+str(R/'src/_julia_ka_env'),'main.jl'],'verification_method':'full_discrete_fields_and_independent_periodic_mass_freeenergy_identity'}
(p/'study_config_reference.json').write_text(json.dumps(config,indent=2)+'\n');(p/'SCOPE.md').write_text('# CHE predeclared periodic conserved-phase study\n\n'+json.dumps(config['input_contract'],indent=2)+'\n\nFiniteallfields,1e-11+1e-11rel independent anddualreferenceacceptance; exactinputprefixandallallocationguards; absoluteperiodicmassdrift<=1e-10. Constantstateanalyticfields andsingleFourierinput included. Independent localpotential identity uses exactcoefficient mathematics, allowing roundoff from sourcebinary2/9 coefficients. No energy-decay theorem or continuous PDE convergence claimed. Native sourcebodyspan retained verbatim; namespaces specialize onlydimensions.\n');shutil.copy2('/tmp/prepare_che.py',p/'study_generator_history.py');print({'prefix':len(fixture),'fields':len(oracle),'guards':32,'artifact_count':len(fixture)+len(oracle)+32})
