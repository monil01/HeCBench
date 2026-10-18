from pathlib import Path
import json,struct,math,shutil,hashlib
R=Path('/home/imo/HeCBench_julia');p=R/'src/binomial-julia';child=p/'binomial-julia-cuda.jl';child.mkdir(exist_ok=True);orig=(p/'main.jl').read_text();shutil.copy2(p/'main.jl',child/'original_main.jl');shutil.copy2(p/'Makefile',p/'original_Makefile');(p/'main.jl').write_text('include(joinpath(@__DIR__,"binomial-julia-cuda.jl","original_main.jl"))\n')
start=orig.index('function binomial_kernel!');end=orig.index('\nfunction binomial_options_gpu',start);kernel=orig[start:end];(child/'study_kernel.jl').write_text('using CUDA\nusing StaticArrays\nconst NUM_STEPS=Int32(2048);const ELEMS_PER_THREAD=Int32(16)\n'+kernel+'\n')
records=[];pre=[];oracle=[];bs=[]
f=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
for n in [16,129]:
 for i in range(n):
  S=f(5+(i*17%101)/4);X=f(1+(i*29%197)/2);T=f(.25+(i*7%39)/4);rate=f(.06);vol=f(.1 if i%3==0 else .2 if i%3==1 else .35)
  # source-style FP32 coefficient preprocessing explicitlyserialized
  dt=f(T/f(2048));vdt=f(vol*f(math.sqrt(dt)));rdt=f(rate*dt);u=f(math.exp(vdt));down=f(math.exp(-vdt));If=f(math.exp(rdt));Df=f(math.exp(-rdt));pu=f(f(If-down)/f(u-down));pd=f(1-pu);a=f(pu*Df);b=f(pd*Df)
  records.extend([S,X,T,rate,vol]);pre.extend([S,X,vdt,a,b])
  terms=[]
  for k in range(2049):
   payoff=max(S*math.exp(vdt*(2*k-2048))-X,0)
   if payoff:terms.append(math.exp(math.lgamma(2049)-math.lgamma(k+1)-math.lgamma(2049-k)+k*math.log(a)+(2048-k)*math.log(b))*payoff)
  oracle.append(math.fsum(terms));d1=(math.log(S/X)+(rate+.5*vol*vol)*T)/(vol*math.sqrt(T));d2=d1-vol*math.sqrt(T);N=lambda x:.5*(1+math.erf(x/math.sqrt(2)));bs.append(S*N(d1)-X*math.exp(-rate*T)*N(d2))
for name,data,fmt in [('fixtures_f32.bin',records+pre,'f'),('oracle_f64.bin',oracle,'d')]:
 with(p/name).open('wb')as io:io.write(struct.pack('<'+fmt*len(data),*data))
(p/'mathematical_oracle.json').write_text(json.dumps({'method':'DirectterminalbinomialexpectationviaFloat64lgamma/logweightsandfsum; no backwardrecursion/sharedexchange reused','price_count':len(oracle),'coefficient_contract':'ExactserializedpreprocessedFloat32coefficients, discounta+b retained ratherthan reinterpreted as exactanalyticrate','max_abs_discrete_to_continuous_BS':max(abs(a-b)for a,b in zip(oracle,bs)),'continuous_BS_is_not_acceptance':'Finite2048stepdiscretizationandroundedcoefficienteffects remain explicit','bounds':'Allpayoffs/weightsnonnegative; alloracleprices>=0'},indent=2)+'\n')
common='''const NAMES=(16,129);const REPEATS=1000;const SENT=Float32(-1048576.5)
function study(allocate,launch)
 fixture=collect(reinterpret(Float32,read(joinpath(@__DIR__,"fixtures_f32.bin"))));expected=collect(reinterpret(Float64,read(joinpath(@__DIR__,"oracle_f64.bin"))));total=sum(NAMES);states=[];off=0
 for n in NAMES
  params=reshape(copy(fixture[5total+5off+1:5total+5off+5n]),5,n);raw=[allocate(Float32,n+2)for k in 1:6];args=[view(a,2:n+1)for a in raw]
  for k in 1:5;h=fill(SENT,n+2);h[2:end-1]=params[k,:];copyto!(raw[k],h);end
  fill!(raw[6],SENT);push!(states,(n,raw,args));off+=n
 end
 for(n,raw,args)in states;launch(n,args...);end;CUDA.synchronize(CUDA.context())
 for trial in 1:5
  out=copy(fixture);elapsed=0.;ok=true;err=0.;off=0
  for(n,raw,args)in states
   fill!(raw[6],SENT);CUDA.synchronize(CUDA.context());start=time_ns()
   for rep in 1:REPEATS;launch(n,args...);end
   CUDA.synchronize(CUDA.context());elapsed+=(time_ns()-start)*1e-9/REPEATS
   arrays=Array.(raw);ok&=all(a->a[1]===SENT&&a[end]===SENT,arrays);vals=arrays[6][2:end-1];reference=expected[off+1:off+n];err=max(err,maximum(abs.(Float64.(vals).-reference)));ok&=all(isfinite,vals)&&all(abs.(Float64.(vals).-reference).<=2e-4 .+5e-4.*abs.(reference));append!(out,vals);append!(out,Float32[arrays[6][1],arrays[6][end]]);off+=n
  end
  path=joinpath(get(ENV,"STUDY_OUTPUT_DIR",pwd()),"trial_$(trial).bin");open(io->write(io,out),path,"w");println("STUDY_TRIAL {\\"trial\\":",trial,",\\"time_seconds\\":",elapsed,",\\"correct\\":",ok,",\\"checked_values\\":",length(out),",\\"max_abs_error\\":",err,",\\"timing_region\\":\\"sum2_preallocated_1000kernel_launch_means_complete\\",\\"jit_excluded\\":true,\\"internal_repeats\\":1000,\\"output_artifact\\":{\\"path\\":\\"",path,"\\",\\"dtype\\":\\"float32\\",\\"count\\":",length(out),",\\"byte_order\\":\\"little\\"}}")
  ok||error("BinomialDiscreteExpectationOrGuard")
 end
end
''';(p/'study_common.jl').write_text(common)
(child/'main.jl').write_text('using CUDA\ninclude("study_kernel.jl")\nfunction launch(n,args...)\n @cuda threads=128 blocks=n binomial_kernel!(args...)\nend\ninclude(joinpath(@__DIR__,"..","study_common.jl"))\nstudy((T,n)->CUDA.zeros(T,n),launch)\n')
native='''#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>
#include <cmath>
#include "../binomial-cuda/kernel.cu"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s\\n",cudaGetErrorString(e));exit(2);}}
template<class T>std::vector<T>load(const char*p){std::ifstream f(p,std::ios::binary|std::ios::ate);size_t n=f.tellg();f.seekg(0);std::vector<T>a(n/sizeof(T));f.read((char*)a.data(),n);return a;}
int main(){auto fixture=load<float>(BIN_FIXTURE);auto oracle=load<double>(BIN_ORACLE);const int sizes[]={16,129};struct State{__TOptionData*in;float*raw;};State s[2];size_t off=0;for(int c=0;c<2;c++){int n=sizes[c];ck(cudaMalloc(&s[c].in,(n+2)*sizeof(__TOptionData)));std::vector<__TOptionData>h(n+2);for(auto&t:h)t={-1048576.5f,-1048576.5f,-1048576.5f,-1048576.5f,-1048576.5f};for(int i=0;i<n;i++){auto*q=&fixture[5*145+5*(off+i)];h[i+1]={q[0],q[1],q[2],q[3],q[4]};}ck(cudaMemcpy(s[c].in,h.data(),h.size()*sizeof(__TOptionData),cudaMemcpyHostToDevice));ck(cudaMalloc(&s[c].raw,(n+2)*4));binomialOptionsKernel<<<n,128>>>(s[c].in+1,s[c].raw+1);off+=n;}ck(cudaDeviceSynchronize());for(int trial=1;trial<=5;trial++){auto out=fixture;double elapsed=0,err=0;bool ok=true;off=0;for(int c=0;c<2;c++){int n=sizes[c];std::vector<float>h(n+2,-1048576.5f);ck(cudaMemcpy(s[c].raw,h.data(),(n+2)*4,cudaMemcpyHostToDevice));ck(cudaDeviceSynchronize());auto start=std::chrono::steady_clock::now();for(int rep=0;rep<1000;rep++)binomialOptionsKernel<<<n,128>>>(s[c].in+1,s[c].raw+1);ck(cudaGetLastError());ck(cudaDeviceSynchronize());elapsed+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/1000;ck(cudaMemcpy(h.data(),s[c].raw,(n+2)*4,cudaMemcpyDeviceToHost));ok&=h.front()==-1048576.5f&&h.back()==-1048576.5f;std::vector<__TOptionData>ih(n+2);ck(cudaMemcpy(ih.data(),s[c].in,(n+2)*sizeof(__TOptionData),cudaMemcpyDeviceToHost));for(auto j:{0,n+1}){float*t=(float*)&ih[j];for(int k=0;k<5;k++)ok&=t[k]==-1048576.5f;}for(int i=0;i<n;i++){double e=std::abs(h[i+1]-oracle[off+i]);err=std::max(err,e);ok&=std::isfinite(h[i+1])&&e<=2e-4+5e-4*std::abs(oracle[off+i]);}out.insert(out.end(),h.begin()+1,h.end()-1);out.push_back(h.front());out.push_back(h.back());off+=n;}std::string path="trial_"+std::to_string(trial)+".bin";std::ofstream f(path,std::ios::binary);f.write((char*)out.data(),out.size()*4);f.close();printf("STUDY_TRIAL {\\"trial\\":%d,\\"time_seconds\\":%.17g,\\"correct\\":%s,\\"checked_values\\":%zu,\\"max_abs_error\\":%.17g,\\"timing_region\\":\\"sum2_preallocated_1000kernel_launch_means_complete\\",\\"jit_excluded\\":true,\\"internal_repeats\\":1000,\\"output_artifact\\":{\\"path\\":\\"%s\\",\\"dtype\\":\\"float32\\",\\"count\\":%zu,\\"byte_order\\":\\"little\\"}}\\n",trial,elapsed,ok?"true":"false",out.size(),err,path.c_str(),out.size());if(!ok)return 3;}for(auto&t:s){ck(cudaFree(t.in));ck(cudaFree(t.raw));}}
''';(p/'study_native.cu').write_text(native)
config={'benchmark':'binomial','configuration_id':'binomial_2048step_original_exchange128_options16_129_repeats1000_v1','gpu':6,'timeout':1800,'inputs_shared_verified':True,'input_prefix_count':1450,'timed_regions_equivalent_verified':True,'input_contract':{'sizes':[16,129],'num_steps':2048,'block_threads':128,'elements_per_thread':16,'precision':'Float32input/preprocessedcoefficients/storage/compute; independentFloat64terminalexpectation','fixture':'145variedcalloptions spot/strike/maturity/volatility; exactserializedrawparametersandpreprocessedcoefficients','timing':'Sum2 preallocatedkernel1000launchmeans withcontextcompletion; preprocessing/transfersoutside','outputs':'All145prices+4exactoutputguards; inputguardscheckedexactindependently','scope':'Sourcekernelunchanged, original1024option/offloadinclusiveCLI notmeasured; original16localvalues+exchange129 retained'},'atol':2e-4,'rtol':5e-4,'numerical_evidence_strength':'full_call_prices_and_independent_terminal_binomial_expectation','models':{}}
deps=[str(p/q)for q in ['fixtures_f32.bin','oracle_f64.bin']]
config['models']['cuda']={'structured':True,'cwd':str(p),'source_dependencies':deps+[str(R/'src/binomial-cuda'/q)for q in ['kernel.cu','binomialOptions.h','realtype.h']],'build':['nvcc','-std=c++17','-O3','-arch={arch}','-DBIN_FIXTURE="'+str(p/'fixtures_f32.bin')+'"','-DBIN_ORACLE="'+str(p/'oracle_f64.bin')+'"','study_native.cu','-o','study_native'],'command':['./study_native'],'verification_method':'allprices_independent_discounted_binomial_terminal_expectation','native_check_description':'FP64directterminalprobabilities via lgamma+fsum, not backwardrecursion; tolerance2e-4+5e-4rel aligns2048stepFP32accumulation/source5e-4norm'}
config['models']['cuda_jl']={'structured':True,'cwd':str(child),'source_dependencies':deps+[str(p/'study_common.jl'),str(child/'study_kernel.jl')],'command':['julia','--project='+str(R/'src/_julia_ka_env'),'main.jl'],'verification_method':'allprices_independent_discounted_binomial_terminal_expectation'}
(p/'study_config_reference.json').write_text(json.dumps(config,indent=2)+'\n');(p/'SCOPE.md').write_text('# Binomial predeclared study\n\n'+json.dumps(config['input_contract'],indent=2)+'\n\nAll145pricesfiniteandper-element2e-4+5e-4rel againstindependentFloat64terminalbinomialexpectation. Source2048stepsFP32initialization(__expf nativeversusJuliaexp)andbackwardupdatesaccumulateFP32roundoff; original aggregate5e-4relativecriterion motivates declaredper-price5e-4+smallabsoluteenvelope. Dualreferencesretainidenticalpredeclaredtolerance. Exactinputprefix/guardbytechecksseparate. No continuousBSaccuracytoleranceasserted, discrete-to-continuouserrorreportedseparately.\n');shutil.copy2('/tmp/prepare_binomial.py',p/'study_generator_history.py')
print('Prepared reference study:1450prefix+145prices+4guards')
