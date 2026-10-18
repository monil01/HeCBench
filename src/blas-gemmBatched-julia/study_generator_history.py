from pathlib import Path
import math,struct,json,shutil,hashlib
R=Path('/home/imo/HeCBench_julia');p=R/'src/blas-gemmBatched-julia';child=p/'blas-gemmBatched-julia-cuda.jl';child.mkdir(exist_ok=True);shutil.copy2(p/'main.jl',child/'original_main.jl');shutil.copy2(p/'Makefile',p/'original_Makefile');(p/'main.jl').write_text('include(joinpath(@__DIR__,"blas-gemmBatched-julia-cuda.jl","original_main.jl"))\n')
fixtures=[];oracle=[];bounds=[];cases=[];U=17
for code,fmt,unit in [(0,'e',2**-11),(1,'f',2**-24),(2,'d',2**-53)]:
 cast=lambda x:struct.unpack('<'+fmt,struct.pack('<'+fmt,x))[0]
 for mode in [0,1]:
  for batch in [3,17]:
   aa=[cast(((i*17+3)%19-9)/(16 if mode==0 else 13)) for i in range(U*U*batch)]
   bb=[cast(((i*11+7)%17-8)/(16 if mode==0 else 11)) for i in range(U*batch)]
   start=len(fixtures);fixtures.extend([code,mode,batch,U,len(aa),len(bb)]+aa+bb);cases.append({'code':code,'mode':mode,'batch':batch,'offset':start,'data_count':len(aa)+len(bb)})
   for m in range(2,U+1):
    for b in range(batch):
     for row in range(U):
      if row>=m:oracle.append(-1024.);bounds.append(0.);continue
      products=[aa[b*U*U+row+k*U]*bb[b*U+k] for k in range(m)];v=math.fsum(products);gamma=(2*m*unit)/(1-2*m*unit);oracle.append(v);bounds.append(gamma*math.fsum(abs(x)for x in products)+4*unit*abs(v))
for name,data in [('fixtures_f64.bin',fixtures),('oracle_f64.bin',oracle),('bounds_f64.bin',bounds)]:
 with(p/name).open('wb')as f:f.write(struct.pack('<'+'d'*len(data),*data))
(p/'fixture_layout.json').write_text(json.dumps({'upper':17,'sizes':list(range(2,18)),'types':['Float16','Float32','Float64'],'cases':cases,'prefix_count':len(fixtures),'payload_count':len(oracle),'oracle':'Independent Float64 math.fsum over exact stored scalar inputs, no cuBLAS/CPUsource checker reuse','bound':'gamma_(2m)(unit)*sum(abs(products))+4unit*abs(expected), unit=2^-11/-24/-53. Padding exact. Source storage/API precision bound, internal vendor accumulation not inferred.','source_alpha':1,'source_beta':0},indent=2)+'\n')
lib='/usr/lib/x86_64-linux-gnu/libcublas.so.12.0.2.224';lt='/usr/lib/x86_64-linux-gnu/libcublasLt.so.12.0.2.224'
# Native adapter retains original source through renamed entry point; invokes same exact three vendor APIs.
native=r'''#define main original_main
#include "../blas-gemmBatched-cuda/main.cu"
#undef main
#include <vector>
#include <fstream>
#include <dlfcn.h>
#include <string>
static void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(e));exit(2);}}
static void cb(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS:%d\n",int(e));exit(3);}}
std::vector<double>load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);size_t n=f.tellg();f.seekg(0);std::vector<double>v(n/8);f.read((char*)v.data(),n);return v;}
static constexpr uintptr_t PGS=0xdecafbad12345678ULL;
template<class T>struct State{int batch;T*ar,*br,*cr;T**apr,**bpr,**cpr;std::vector<T>ah,bh;std::vector<T*>ap,bp,cp;State(int n,const double*a,const double*b):batch(n){int na=17*17*n,nb=17*n;ah.resize(na+2,T(-1024.f));bh.resize(nb+2,T(-1024.f));for(int i=0;i<na;i++)ah[i+1]=T(a[i]);for(int i=0;i<nb;i++)bh[i+1]=T(b[i]);ck(cudaMalloc(&ar,(na+2)*sizeof(T)));ck(cudaMalloc(&br,(nb+2)*sizeof(T)));ck(cudaMalloc(&cr,(nb+2)*sizeof(T)));ck(cudaMemcpy(ar,ah.data(),ah.size()*sizeof(T),cudaMemcpyHostToDevice));ck(cudaMemcpy(br,bh.data(),bh.size()*sizeof(T),cudaMemcpyHostToDevice));for(auto*list:{&ap,&bp,&cp})list->resize(n+2,reinterpret_cast<T*>(PGS));for(int j=0;j<n;j++){ap[j+1]=ar+1+j*289;bp[j+1]=br+1+j*17;cp[j+1]=cr+1+j*17;}ck(cudaMalloc(&apr,(n+2)*sizeof(T*)));ck(cudaMalloc(&bpr,(n+2)*sizeof(T*)));ck(cudaMalloc(&cpr,(n+2)*sizeof(T*)));ck(cudaMemcpy(apr,ap.data(),ap.size()*sizeof(T*),cudaMemcpyHostToDevice));ck(cudaMemcpy(bpr,bp.data(),bp.size()*sizeof(T*),cudaMemcpyHostToDevice));ck(cudaMemcpy(cpr,cp.data(),cp.size()*sizeof(T*),cudaMemcpyHostToDevice));}
void call(cublasHandle_t h,int m){T a=T(1.f),b=T(0.f);if constexpr(std::is_same_v<T,__half>)cb(cublasHgemmBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,m,1,m,&a,(const T**)(apr+1),17,(const T**)(bpr+1),17,&b,cpr+1,17,batch));else if constexpr(std::is_same_v<T,float>)cb(cublasSgemmBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,m,1,m,&a,(const T**)(apr+1),17,(const T**)(bpr+1),17,&b,cpr+1,17,batch));else cb(cublasDgemmBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,m,1,m,&a,(const T**)(apr+1),17,(const T**)(bpr+1),17,&b,cpr+1,17,batch));}
double run(cublasHandle_t h,std::vector<double>&out,const std::vector<double>&oracle,const std::vector<double>&bound,size_t&oi,bool&ok,double&err){double elapsed=0;int nb=17*batch;for(int m=2;m<=17;m++){std::vector<T>ch(nb+2,T(-1024.f));ck(cudaMemcpy(cr,ch.data(),ch.size()*sizeof(T),cudaMemcpyHostToDevice));for(int rep=0;rep<=10;rep++){ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();call(h,m);ck(cudaDeviceSynchronize());double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();if(rep)elapsed+=dt/10;}ck(cudaMemcpy(ch.data(),cr,ch.size()*sizeof(T),cudaMemcpyDeviceToHost));ok&=double(ch.front())==-1024.&&double(ch.back())==-1024.;for(int j=0;j<nb;j++){double val=double(ch[j+1]),e=std::abs(val-oracle[oi]);ok&=std::isfinite(val)&&e<=bound[oi]+1e-30;if(j%17>=m)ok&=val==-1024.;else err=std::max(err,e);out.push_back(val);oi++;}out.push_back(double(ch.front()));out.push_back(double(ch.back()));out.push_back(0.);}
std::vector<T>ha(ah.size()),hb(bh.size());ck(cudaMemcpy(ha.data(),ar,ha.size()*sizeof(T),cudaMemcpyDeviceToHost));ck(cudaMemcpy(hb.data(),br,hb.size()*sizeof(T),cudaMemcpyDeviceToHost));ok&=memcmp(ha.data(),ah.data(),ha.size()*sizeof(T))==0&&memcmp(hb.data(),bh.data(),hb.size()*sizeof(T))==0;for(auto pair:{std::make_pair(apr,&ap),std::make_pair(bpr,&bp),std::make_pair(cpr,&cp)}){std::vector<T*>tmp(batch+2);ck(cudaMemcpy(tmp.data(),pair.first,tmp.size()*sizeof(T*),cudaMemcpyDeviceToHost));ok&=tmp==*pair.second;}return elapsed;}
void free(){for(auto*x:{ar,br,cr})ck(cudaFree(x));for(auto*x:{apr,bpr,cpr})ck(cudaFree(x));}};
int main(){auto fixture=load(BATCH_FIXTURE),oracle=load(BATCH_ORACLE),bounds=load(BATCH_BOUNDS);cublasHandle_t h;cb(cublasCreate(&h));cb(cublasSetStream(h,0));int version=0;cb(cublasGetVersion(h,&version));Dl_info info;dladdr(dlsym(RTLD_DEFAULT,"cublasCreate_v2"),&info);printf("STUDY_LIBRARY path=%s version=%d\n",info.dli_fname,version);std::vector<State<__half>*>hs;std::vector<State<float>*>ss;std::vector<State<double>*>ds;size_t off=0;for(int code=0;code<3;code++)for(int mode=0;mode<2;mode++)for(int n:{3,17}){int na=289*n,nb=17*n;auto*a=&fixture[off+6];auto*b=a+na;if(code==0)hs.push_back(new State<__half>(n,a,b));if(code==1)ss.push_back(new State<float>(n,a,b));if(code==2)ds.push_back(new State<double>(n,a,b));off+=6+na+nb;}for(auto*s:hs)s->call(h,17);for(auto*s:ss)s->call(h,17);for(auto*s:ds)s->call(h,17);ck(cudaDeviceSynchronize());for(int trial=1;trial<=5;trial++){auto out=fixture;size_t oi=0;bool ok=true;double elapsed=0,err=0;for(auto*s:hs)elapsed+=s->run(h,out,oracle,bounds,oi,ok,err);for(auto*s:ss)elapsed+=s->run(h,out,oracle,bounds,oi,ok,err);for(auto*s:ds)elapsed+=s->run(h,out,oracle,bounds,oi,ok,err);std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)out.data(),out.size()*8);f.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"timing_region\":\"sum192_batchedGEMV_10callmeans_skip_internal0_contextcomplete\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,elapsed,ok?"true":"false",out.size(),err,path.c_str(),out.size());if(!ok)return 4;}for(auto*s:hs){s->free();delete s;}for(auto*s:ss){s->free();delete s;}for(auto*s:ds){s->free();delete s;}cb(cublasDestroy(h));}
''';(p/'study_native.cu').write_text(native)
common=r'''using Libdl
const STUDY_LIB="/usr/lib/x86_64-linux-gnu/libcublas.so.12.0.2.224"
const PGS=UInt64(0xdecafbad12345678)
for T in (Float16,Float32,Float64)
 @eval function gemmcall(fn,h,m,alpha::Ref{$T},ap,bp,beta::Ref{$T},cp,batch)
  ccall(fn,Cint,(Ptr{Cvoid},Cint,Cint,Cint,Cint,Cint,Ref{$T},CuPtr{Cvoid},Cint,CuPtr{Cvoid},Cint,Ref{$T},CuPtr{Cvoid},Cint,Cint),h,0,0,m,1,m,alpha,pointer(ap,2),17,pointer(bp,2),17,beta,pointer(cp,2),17,batch)
 end
end
function study(allocate)
 fixture=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixtures_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));bounds=collect(reinterpret(Float64,read(joinpath(@__DIR__,"bounds_f64.bin"))))
 lib=Libdl.dlopen(STUDY_LIB);href=Ref{Ptr{Cvoid}}();ccall(Libdl.dlsym(lib,:cublasCreate_v2),Cint,(Ref{Ptr{Cvoid}},),href)==0||error("CreateHandle")
 ccall(Libdl.dlsym(lib,:cublasSetStream_v2),Cint,(Ptr{Cvoid},Ptr{Cvoid}),href[],Ptr{Cvoid}(CUDA.stream().handle))==0||error("BindStream")
 ver=Ref{Cint}();ccall(Libdl.dlsym(lib,:cublasGetVersion_v2),Cint,(Ptr{Cvoid},Ref{Cint}),href[],ver)==0||error("LibraryVersion");println("STUDY_LIBRARY path=",realpath(STUDY_LIB)," version=",ver[]," task_stream_bound=true")
 states=[];off=0
 for(T,symbol)in ((Float16,:cublasHgemmBatched),(Float32,:cublasSgemmBatched),(Float64,:cublasDgemmBatched)),mode in 0:1,n in (3,17)
  na=289n;nb=17n;ah=vcat(T(-1024),T.(fixture[off+7:off+6+na]),T(-1024));bh=vcat(T(-1024),T.(fixture[off+7+na:off+6+na+nb]),T(-1024));a=allocate(T,na+2);b=allocate(T,nb+2);c=allocate(T,nb+2);copyto!(a,ah);copyto!(b,bh)
  ph=[vcat(PGS,UInt64[UInt64(pointer(x,2+(j-1)*stride)) for j in 1:n],PGS) for(x,stride)in ((a,289),(b,17),(c,17))];ap=[allocate(UInt64,n+2) for k in 1:3];for k in 1:3;copyto!(ap[k],ph[k]);end
  push!(states,(T,n,a,b,c,ap,ah,bh,ph,Libdl.dlsym(lib,symbol)));off+=6+na+nb
 end
 function call(c,m)
  T,n,a,b,out,ap,ah,bh,ph,fn=c;gemmcall(fn,href[],Cint(m),Ref{T}(one(T)),ap[1],ap[2],Ref{T}(zero(T)),ap[3],Cint(n))==0||error("GemmBatchedStatus")
 end
 for c in states;call(c,17);end;CUDA.synchronize(CUDA.context())
 for trial in 1:5
  out=copy(fixture);oi=0;elapsed=0.;ok=true;err=0.
  for c in states
   T,n,a,b,dev,ap,ah,bh,ph,fn=c
   for m in 2:17
    fill!(dev,T(-1024))
    for rep in 0:10
     CUDA.synchronize(CUDA.context());start=time_ns();call(c,m);CUDA.synchronize(CUDA.context());dt=(time_ns()-start)*1e-9;rep>0&&(elapsed+=dt/10)
    end
    got=Array(dev);ok&=isequal(got[1],T(-1024))&&isequal(got[end],T(-1024))
    for j in 1:17n
     v=Float64(got[j+1]);e=abs(v-oracle[oi+j]);ok&=isfinite(v)&&e<=bounds[oi+j]+1e-30
     if mod(j-1,17)>=m;ok&=isequal(got[j+1],T(-1024));else;err=max(err,e);end
     push!(out,v)
    end
    append!(out,Float64[got[1],got[end],0]);oi+=17n
   end
   ok&=reinterpret(UInt8,Array(a))==reinterpret(UInt8,ah)&&reinterpret(UInt8,Array(b))==reinterpret(UInt8,bh)&&all(Array(ap[k])==ph[k] for k in 1:3)
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w")
  println("STUDY_TRIAL {\"trial\":",trial,",\"time_seconds\":",elapsed,",\"correct\":",ok,",\"checked_values\":",length(out),",\"max_abs_error\":",err,",\"timing_region\":\"sum192_batchedGEMV_10callmeans_skip_internal0_contextcomplete\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"",path,"\",\"dtype\":\"float64\",\"count\":",length(out),",\"byte_order\":\"little\"}}")
  ok||error("BatchedGEMVFullPayloadPaddingPointersAndDotOracle")
 end
 ccall(Libdl.dlsym(lib,:cublasDestroy_v2),Cint,(Ptr{Cvoid},),href[])==0||error("DestroyHandle")
end
''';(p/'study_common.jl').write_text(common)
(child/'main.jl').write_text('using CUDA\ninclude(joinpath(@__DIR__,"..","study_common.jl"))\nstudy((T,n)->CUDA.zeros(T,n))\n')
preload='LD_PRELOAD='+lib+':'+lt
cfg={'benchmark':'blas-gemmBatched','configuration_id':'gemmBatched_HSD_all_sizes2_17_batches3_17_twofixtures_pinned120_v1','gpu':6,'timeout':1200,'inputs_shared_verified':True,'input_prefix_count':len(fixtures),'timed_regions_equivalent_verified':True,'input_contract':{'upper_leading_dimension':17,'sizes':list(range(2,18)),'batches':[3,17],'types':['Float16','Float32','Float64'],'fixtures':['Signedexactdyadic','Signedroundedrational'],'alpha':1,'beta':0,'transpose':['N','N'],'gemm_n':1,'timing':'Sum192 size/type/batch/fixture points, each mean10calls rep1..10 with rep0excluded; contextcomplete eachcall. Allocation/inputsetup/readbackoutside. Fiveoutertrials discardonlyfirstoutertiming.','output':'Full17*batchoutputeachsize including exactinactivepadding andouterguardvalues plusstatus; allinputandpointerlistscheckedbitexact','input_encoding':'Float64serializedactualFloat16/Float32/Float64values exactlyrepresentable inFloat64; exactbyteprefix acrossall4','independent_acceptance':'Float64fsum dots, boundgamma_2m(typeunit)*sumabsproducts+4unit*absoracle, paddingandguardsbitexact','library':'Pinned+preloaded cuBLAS/Lt12.0.2.224 all4, actualruntimepath/version printed; explicitJuliahandletaskstreambinding/contextcompletion','scope':'Zeroauthoredkernels; KA/JACCframeworkallocation withCUDAvendorinterop. Originalupper100/batch25000CLIunmeasured.'},'atol':0,'rtol':0,'numerical_evidence_strength':'full_padded_batched_GEMV_all3precisions_and_independent_dot_roundoff_bounds','models':{}}
deps=[str(p/q)for q in ['fixtures_f64.bin','oracle_f64.bin','bounds_f64.bin']]+[lib,lt]
cfg['models']['cuda']={'structured':True,'cwd':str(p),'source_dependencies':deps+[str(R/'src/blas-gemmBatched-cuda'/q)for q in ['main.cu','reference.h']],'build':['nvcc','-std=c++17','-O3','-arch={arch}','-DBATCH_FIXTURE="'+str(p/'fixtures_f64.bin')+'"','-DBATCH_ORACLE="'+str(p/'oracle_f64.bin')+'"','-DBATCH_BOUNDS="'+str(p/'bounds_f64.bin')+'"','study_native.cu',lib,'-ldl','-o','study_native'],'command':['env',preload,'./study_native'],'verification_method':'all3precisionfulloutputs_dotbounds_exactpadding_inputs_and_pointers','native_check_description':'UnchangedsourceH/S/DgemmBatchedcallABI, guardednamedharness; independent dotoracleavoids uninitializedC read inoriginalCPU beta0expression.'}
cfg['models']['cuda_jl']={'structured':True,'cwd':str(child),'source_dependencies':deps+[str(p/'study_common.jl')],'command':['env',preload,'julia','--project='+str(R/'src/_julia_ka_env'),'main.jl'],'verification_method':'all3precisionfulloutputs_dotbounds_exactpadding_inputs_and_pointers'}
(p/'study_config_reference.json').write_text(json.dumps(cfg,indent=2)+'\n');(p/'SCOPE.md').write_text('# BatchedGEMM predeclared vendor-interoperability study\n\n'+json.dumps(cfg['input_contract'],indent=2)+'\n\nCross-model actualoutputs/inputprefixexact atol=rtol=0. Independent numerical acceptance uses standardroundoffdotbound bystorage/APIprecision; internalcuBLASaccumulationprecisionnotinferred. Allguards/padding/input/pointerlistbytescheckedexact. H/S/D cases retained; original fullCLI notcertified. Runtime-librarypath/version/hash evidence mandatory, genericlinkflag alone insufficient.\n');shutil.copy2('/tmp/prepare_gemmBatched.py',p/'study_generator_history.py');print({'prefix':len(fixtures),'output_payload':len(oracle),'artifact_count':len(fixtures)+len(oracle)+12*16*3})
