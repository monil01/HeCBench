from pathlib import Path
import json,struct,math,shutil,hashlib
R=Path('/home/imo/HeCBench_julia');p=R/'src/conversion-julia';d=p/'conversion-julia-cuda.jl';d.mkdir(exist_ok=True);shutil.copy2(p/'main.jl',d/'original_main.jl');shutil.copy2(p/'Makefile',p/'original_Makefile');(p/'main.jl').write_text('include(joinpath(@__DIR__,"conversion-julia-cuda.jl","main.jl"))\n')
# Codes BF16=1,F16=2,F32=3,I32=4,I8=5,U8=6; Python standard IEEE bit encodings.
pairs=[(s,t)for s,ts in [(1,[2,3,4,5,6]),(2,[2,3,4,5,6]),(3,[3,2,4,5,6]),(4,[4,3,2,5,6]),(5,[4,3,2,5,6]),(6,[4,3,2,5,6])]for t in ts]
def encode(x,c):
 if c==1:
  b=struct.unpack('<I',struct.pack('<f',x))[0];return (b+0x7fff+((b>>16)&1))>>16
 if c==2:return struct.unpack('<H',struct.pack('<e',x))[0]
 if c==3:return struct.unpack('<I',struct.pack('<f',x))[0]
 return int(x)% (1<<(32 if c==4 else 8))
def decode(b,c):
 if c==1:return struct.unpack('<f',struct.pack('<I',b<<16))[0]
 if c==2:return struct.unpack('<e',struct.pack('<H',b))[0]
 if c==3:return struct.unpack('<f',struct.pack('<I',b))[0]
 if c==4:return b-(1<<32)if b>=(1<<31)else b
 if c==5:return b-256 if b>=128 else b
 return b
fixture=[];oracle=[];records=[]
for s,t in pairs:
 for n in [17,257]:
  vals=([0.,-0.,.5,1.5,2.5,3.75,15.9375,31.5,63.875,126.5,127.,1+2**-11,1+3*2**-11,2**-24,2**-14,1+2**-8,1+3*2**-8]if s<=3 else([-32768,-1025,-257,-129,-128,-1,0,1,127,128,255,256,1023,1025,2049,32767]if s==4 else([-128,-127,-1,0,1,63,126,127]if s==5 else[0,1,127,128,129,254,255])))
  bits=[encode(vals[i%len(vals)],s)for i in range(n)];expected=[encode(decode(b,s),t)for b in bits];fixture += [n,s,t]+bits;oracle +=expected;records.append({'n':n,'source_code':s,'destination_code':t})
for name,v in [('fixtures_f64.bin',fixture),('oracle_f64.bin',oracle)]: (p/name).write_bytes(struct.pack('<'+'d'*len(v),*v))
(p/'independent_oracle.json').write_text(json.dumps({'method':'Independent Python IEEE binary packing: BF16 round-to-nearest-even via high-bit integer rounding, IEEE half via struct e, float32 via struct f; integer truncation/modulo fixed widths, no GPU cast reused','cases':records,'input_values':'Finite floating values in0..127 valid for all float-to-integer targets, signed zero/halfway/subnormal values; fullsigned8/unsigned8 extrema and signed32 negatives/narrowing wrap','scope':'Float->integer out-of-range/NaN undefined or implementation-defined cases excluded; char source asserts signed8 on this host. Signed integer narrowing measured modulo2^width implementation policy.'},indent=2)+'\n')
(p/'study_common.jl').write_text('''const TYPES=(CUDA.CUDACore.BFloat16,Float16,Float32,Int32,Int8,UInt8)
const UTYPES=(UInt16,UInt16,UInt32,UInt32,UInt8,UInt8)
@inline cast_value(::Type{T},x::Integer) where {T<:Integer}=x%T
@inline cast_value(::Type{T},x::AbstractFloat) where {T<:Integer}=unsafe_trunc(T,x)
@inline cast_value(::Type{T},x) where {T<:AbstractFloat}=T(x)
function study(allocate,launch)
 fixture=collect(reinterpret(Float64,read(joinpath(@__DIR__,"fixtures_f64.bin"))));oracle=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));states=[];off=0
 while off<length(fixture)
  n,s,t=Int.(fixture[off+1:off+3]);Ts=TYPES[s];Td=TYPES[t];Us=UTYPES[s];Ud=UTYPES[t];bits=Us.(fixture[off+4:off+3+n]);input=collect(reinterpret(Ts,bits));guardS=reinterpret(Ts,Us(0xa5a5a5a5%Us));guardD=reinterpret(Td,Ud(0xa5a5a5a5%Ud));rawin=vcat(guardS,input,guardS);a=allocate(Ts,n+2);b=allocate(Td,n+2);copyto!(a,rawin);push!(states,(n,s,t,a,b,rawin,guardD));off+=3+n
 end
 function run(c)
  n,s,t,a,b,rawin,g=c;fill!(b,g);CUDA.synchronize(CUDA.context());start=time_ns()
  for rep in 1:10;launch(view(b,2:n+1),view(a,2:n+1),Int32(n));end
  CUDA.synchronize(CUDA.context());elapsed=(time_ns()-start)*1e-9/10;got=Array(b);bits=collect(reinterpret(UTYPES[t],got));ok=reinterpret(UInt8,Array(a))==reinterpret(UInt8,rawin)&&isequal(got[1],g)&&isequal(got[end],g)&&all(isfinite,Float64.(got[2:end-1]));return Float64.(bits[2:end-1]),Float64[bits[1],bits[end]],elapsed,ok
 end
 for c in states;run(c);end
 for trial in 1:5
  out=copy(fixture);oi=0;elapsed=0.;ok=true
  for c in states
   vals,guards,dt,pass=run(c);ok&=pass&&vals==oracle[oi+1:oi+length(vals)];oi+=length(vals);append!(out,vals);append!(out,guards);elapsed+=dt
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_TRIAL {\\"trial\\":",trial,",\\"time_seconds\\":",elapsed,",\\"correct\\":",ok,",\\"checked_values\\":",length(out),",\\"max_abs_error\\":0,\\"timing_region\\":\\"sum60_typed_conversion_10launch_means_complete\\",\\"jit_excluded\\":true,\\"internal_repeats\\":10,\\"output_artifact\\":{\\"path\\":\\"",path,"\\",\\"dtype\\":\\"float64\\",\\"count\\":",length(out),",\\"byte_order\\":\\"little\\"}}")
  ok||error("All30TypedConversionsBitPatternsInputAndGuards")
 end
end
''')
(d/'main.jl').write_text('''using CUDA
function convert_kernel!(dst,src,n)
 i=Int32((blockIdx().x-1)*blockDim().x+threadIdx().x)
 if i<=n;@inbounds dst[i]=cast_value(eltype(dst),src[i]);end
 return
end
function launch(dst,src,n)
 threads=min(Int(n),256)
 @cuda threads=threads blocks=cld(Int(n),threads) convert_kernel!(dst,src,n)
end
include(joinpath(@__DIR__,"..","study_common.jl"))
study((T,n)->CUDA.zeros(T,n),launch)
''')
native=r'''#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <cstring>
#include <cstdint>
#include <type_traits>
#define main original_undefined_input_main
#include "../conversion-cuda/main.cu"
#undef main
static_assert(std::is_signed<char>::value,"Sourcecharpolicy");
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
std::vector<double>load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);size_t n=f.tellg();f.seekg(0);std::vector<double>v(n/8);f.read((char*)v.data(),n);return v;}
struct Case{int n,s,t;std::vector<double>input;};
template<class D,class S>void run(const Case&c,std::vector<double>&out,double&elapsed,bool&ok,const std::vector<double>&oracle,size_t&oi){int n=c.n;std::vector<S>h(n+2);memset(h.data(),0xa5,h.size()*sizeof(S));for(int i=0;i<n;i++){uint32_t b=uint32_t(c.input[i]);memcpy(&h[i+1],&b,sizeof(S));}S*a;D*b;ck(cudaMalloc(&a,h.size()*sizeof(S)));ck(cudaMalloc(&b,(n+2)*sizeof(D)));ck(cudaMemcpy(a,h.data(),h.size()*sizeof(S),cudaMemcpyHostToDevice));ck(cudaMemset(b,0xa5,(n+2)*sizeof(D)));int threads=std::min(n,256);cvt<D,S><<<(n+threads-1)/threads,threads>>>(b+1,a+1,n);ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int rep=0;rep<10;rep++)cvt<D,S><<<(n+threads-1)/threads,threads>>>(b+1,a+1,n);ck(cudaDeviceSynchronize());elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/10;ck(cudaGetLastError());std::vector<D>got(n+2);std::vector<S>back(n+2);ck(cudaMemcpy(got.data(),b,got.size()*sizeof(D),cudaMemcpyDeviceToHost));ck(cudaMemcpy(back.data(),a,h.size()*sizeof(S),cudaMemcpyDeviceToHost));ok&=memcmp(h.data(),back.data(),h.size()*sizeof(S))==0;for(int i=1;i<=n;i++){uint32_t bits=0;memcpy(&bits,&got[i],sizeof(D));ok&=double(bits)==oracle[oi++];out.push_back(double(bits));}for(int i:{0,n+1}){uint32_t bits=0,expected=0xa5a5a5a5;memcpy(&bits,&got[i],sizeof(D));ok&=memcmp(&bits,&expected,sizeof(D))==0;out.push_back(double(bits));}ck(cudaFree(a));ck(cudaFree(b));}
template<class S>void dest(const Case&c,std::vector<double>&out,double&t,bool&ok,const std::vector<double>&o,size_t&oi){switch(c.t){case 2:run<half,S>(c,out,t,ok,o,oi);break;case 3:run<float,S>(c,out,t,ok,o,oi);break;case 4:run<int,S>(c,out,t,ok,o,oi);break;case 5:run<char,S>(c,out,t,ok,o,oi);break;case 6:run<uchar,S>(c,out,t,ok,o,oi);}}
void dispatch(const Case&c,std::vector<double>&out,double&t,bool&ok,const std::vector<double>&o,size_t&oi){switch(c.s){case 1:dest<__nv_bfloat16>(c,out,t,ok,o,oi);break;case 2:dest<half>(c,out,t,ok,o,oi);break;case 3:dest<float>(c,out,t,ok,o,oi);break;case 4:dest<int>(c,out,t,ok,o,oi);break;case 5:dest<char>(c,out,t,ok,o,oi);break;case 6:dest<uchar>(c,out,t,ok,o,oi);}}
int main(){auto f=load(CVT_FIXTURE),o=load(CVT_ORACLE);std::vector<Case>cases;size_t i=0;while(i<f.size()){Case c;c.n=int(f[i]);c.s=int(f[i+1]);c.t=int(f[i+2]);c.input.assign(f.begin()+i+3,f.begin()+i+3+c.n);cases.push_back(c);i+=3+c.n;}std::vector<double>dummy;double tm=0;bool pass=true;size_t oi=0;for(auto&c:cases)dispatch(c,dummy,tm,pass,o,oi);for(int trial=1;trial<=5;trial++){auto out=f;double t=0;bool ok=true;size_t oi=0;for(auto&c:cases)dispatch(c,out,t,ok,o,oi);std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream g(path,std::ios::binary);g.write((char*)out.data(),out.size()*8);g.close();printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":0,\"timing_region\":\"sum60_typed_conversion_10launch_means_complete\",\"jit_excluded\":true,\"internal_repeats\":10,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,t,ok?"true":"false",out.size(),path.c_str(),out.size());if(!ok)return 3;}}
''';(p/'study_native.cu').write_text(native)
# Note native includes a percase warmup inside outertrials; outside ROI. Julia run warmup only outside outertrials. Consistent timedlaunches only; disclose setupdifference.
c={'benchmark':'conversion','configuration_id':'conversion_all30_17_257_exactbits_10launch_v1','gpu':6,'timeout':900,'inputs_shared_verified':True,'input_prefix_count':len(fixture),'timed_regions_equivalent_verified':True,'atol':0.,'rtol':0.,'numerical_evidence_strength':'all30_conversion_raw_bitpatterns_independent_IEEE_integer_oracle','input_contract':{'sizes':[17,257],'typecodes':{'1':'BF16','2':'Float16','3':'Float32','4':'Int32','5':'Int8','6':'UInt8'},'typed_pairs':pairs,'inputs':'Fullyinitializedfiniteinrangefloatstoallintegers; half/BF16roundingties/subnormals, signedzero, signed/unsigned8extrema, signed32narrowingwrap','outputs':'ExactrawbitpatternalltypedoutputsencodedFloat64; exactsourceinputprefixand4guardspercase(selfchecksource2,serializeddestination2)','timing':'Sum60cases10launchmeans, sourcegridmin(n,256), fullcontextcompletion, allocation/reset/readbackoutside','scope':'All30nativecastpairs, notoriginaluninitializedinputCLI; nooutofrangefloat-to-integer/NaNclaim; narrower signedintegerconversionimplementationpolicy explicitlypinned'},'models':{}}
deps=[str(p/q)for q in ['fixtures_f64.bin','oracle_f64.bin']];c['models']['cuda']={'structured':True,'cwd':str(p),'source_dependencies':deps+[str(R/'src/conversion-cuda/main.cu')],'build':['nvcc','-std=c++17','-O3','-arch={arch}','-DCVT_FIXTURE="'+str(p/'fixtures_f64.bin')+'"','-DCVT_ORACLE="'+str(p/'oracle_f64.bin')+'"','study_native.cu','-o','study_native'],'command':['./study_native'],'verification_method':'all30exactIEEEbits_narrowing_sourceguards','native_check_description':'Unchangedoriginalcvt instantiatedall30pairs; independentPythonbitpatternoracle allvalues'};c['models']['cuda_jl']={'structured':True,'cwd':str(d),'source_dependencies':deps+[str(p/'study_common.jl')],'command':['julia','--project='+str(R/'src/_julia_ka_env'),'main.jl'],'verification_method':'all30exactIEEEbits_narrowing_sourceguards'};(p/'study_config_reference.json').write_text(json.dumps(c,indent=2)+'\n');(p/'SCOPE.md').write_text('# Thirty typed conversions\n\n'+json.dumps(c['input_contract'],indent=2)+'\n\nAcceptance predeclared: zero-tolerance raw outputbits, inputbits and guards; independent Python IEEEpacking/integertruncation-modulo oracle. Original reads uninitializedallocation, namedfixtures initializeall. No scientificmodel or exceptional/out-of-rangefloatcastclaim. Native percase extra untimedwarmup before each ROI, Julia warms allspecializations before5outertrials; 10 timedcalls equivalent, setup excluded.\n');shutil.copy2('/tmp/prepare_conversion.py',p/'study_generator_history.py');print({'prefix':len(fixture),'outputs':len(oracle),'artifact_count':len(fixture)+len(oracle)+120})
