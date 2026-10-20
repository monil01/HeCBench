// Source-CPU recurrence crosscheck and exact ABI/input serializer, not an independent folding algorithm.
#define FLOAT
#include "../prna-cuda/prna.cu"
#include <vector>
#include <fstream>
#include <string>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <iostream>
template<class T>void save(const std::string&name,const std::vector<T>&a){std::ofstream f(name,std::ios::binary);f.write((const char*)a.data(),a.size()*sizeof(T));assert(f);}
struct State{int n;std::vector<float>v,w5,w3,z,yl,y,wq,w,wl,xl,x;State(int m):n(m),v(m*m,INFINITY),w5(m+1,INFINITY),w3(m+1,INFINITY),z(m*m,INFINITY),yl(m*m,INFINITY),y(m*m,INFINITY),wq(m*(m-1)/2,INFINITY),w(2*m,INFINITY),wl(2*m,INFINITY),xl(2*m,INFINITY),x(5*m,INFINITY){}std::vector<float>snapshot(){std::vector<float>a;for(auto p:{&v,&w5,&w3,&z,&yl,&y,&wq,&w,&wl,&xl,&x})a.insert(a.end(),p->begin(),p->end());return a;}};
int main(int argc,char**argv){assert(argc==3);param par{};param_read_from_text(argv[1],&par,0,0);std::vector<uint32_t>bits(sizeof(par)/4);static_assert(sizeof(par)%4==0);memcpy(bits.data(),&par,sizeof(par));save("params_u32.bin",bits);
std::ofstream layout("parameter_layout.json");layout<<"{\"size_bytes\":"<<sizeof(par)<<",\"word_bytes\":4,\"NBASE\":5,\"fields\":{";bool first=true;
#define FIELD(name) do{if(!first)layout<<",";first=false;layout<<"\"" #name "\":{\"offset_bytes\":"<<offsetof(param,name)<<",\"size_bytes\":"<<sizeof(par.name)<<"}";}while(0)
FIELD(use_dna_params);FIELD(use_enthalpy_params);FIELD(temperature);FIELD(coaxial);FIELD(coaxstack);FIELD(stack);FIELD(tstack);FIELD(tstackcoax);FIELD(tstackh);FIELD(tstacki);FIELD(tstacki1n);FIELD(tstacki23);FIELD(tstackm);FIELD(internal_loop_initiation);FIELD(bulge_loop_initiation);FIELD(hairpin_loop_initiation);FIELD(Extrapolation_for_large_loops);FIELD(maximum_correction);FIELD(fm_array_first_element);FIELD(a);FIELD(multibranched_loop_offset);FIELD(b);FIELD(multibranched_loop_per_nuc_penalty);FIELD(c);FIELD(multibranched_loop_helix_penalty);FIELD(a_2c);FIELD(a_2b_2c);FIELD(terminal_AU_penalty);FIELD(bonus_for_GGG_hairpin);FIELD(c_hairpin_slope);FIELD(c_hairpin_intercept);FIELD(c_hairpin_of_3);FIELD(Bonus_for_Single_C_bulges_adjacent_to_C);FIELD(ntriloop);FIELD(triloop);FIELD(ntloop);FIELD(tloop);FIELD(nhexaloop);FIELD(hexaloop);FIELD(dangle_3p);FIELD(dangle_5p);FIELD(int11);FIELD(int22);FIELD(int21);layout<<"}}\n";layout.close();
char*provided=sequence(argv[2]);std::vector<std::string>texts={provided,"GCAAAGC","AAAAAAAAA","GcAAAgC","GGGAAACCCGGGAAACCCGGGAAACCC"};free(provided);std::vector<int32_t> seqs{int32_t(texts.size())},pairs,shape;std::vector<float> gold,trace;std::ofstream summary("CPU_reference_summary.json");summary<<"{\"crosscheck_kind\":\"source_serial_CPU_not_independent_algorithm\",\"initialization\":\"allstate_payloads_positive_infinity_before_sourceinit, parameter_struct_value_initialized_before_parser\",\"cases\":[";
for(size_t q=0;q<texts.size();q++){auto text=texts[q];int n=text.size();std::vector<base_t>s(n);sequence_from_string(s.data(),text.c_str());int*raw=generate_bcp(text.c_str());std::vector<int>bcp(raw,raw+n*(n-1)/2);free(raw);seqs.push_back(n);for(auto x:s)seqs.push_back(x);pairs.insert(pairs.end(),bcp.begin(),bcp.end());shape.push_back(n);State a(n);auto capture=[&](){if(q==4){auto b=a.snapshot();trace.insert(trace.end(),b.begin(),b.end());}};init_w5_and_w3(n,a.w5.data()+1,a.w3.data());capture();for(int d=0;d<n-1;d++){
calc_hairpin_stack_exterior_multibranch(d,n,s.data(),bcp.data(),a.v.data(),a.x.data(),a.w5.data()+1,a.w3.data(),&par);capture();
calc_internal(d,n,s.data(),bcp.data(),a.v.data(),&par);capture();
calc_coaxial(d,n,s.data(),bcp.data(),a.v.data(),a.y.data(),a.w5.data()+1,a.w3.data(),&par);capture();
calc_wl(d,n,s.data(),bcp.data(),a.v.data(),a.z.data(),a.wq.data(),a.w.data(),a.wl.data(),&par);capture();
calc_xl(d,n,a.z.data(),a.yl.data(),a.xl.data());capture();
calc_z(d,n,s.data(),bcp.data(),a.v.data(),a.z.data(),a.xl.data(),a.wq.data(),&par);capture();
calc_x(d,n,a.yl.data(),a.y.data(),a.w.data(),a.wl.data(),a.xl.data(),a.x.data(),&par);capture();
calc_w5_and_w3(d,n,a.w5.data()+1,a.w3.data(),a.wq.data());capture();}
auto b=a.snapshot();gold.insert(gold.end(),b.begin(),b.end());assert(std::isfinite(a.w3[0]));double massmax=0,pmax=0;for(int i=0;i<n;i++){double mass=0;for(int j=0;j<n;j++){double prob=can_pair(i,j,n,bcp.data())?exp(-double(a.v[i*n+j]+a.v[j*n+i]-a.w3[0])):0;assert(std::isfinite(prob));mass+=prob;pmax=std::max(pmax,prob);}massmax=std::max(massmax,mass);}if(q)summary<<",";summary<<"{\"n\":"<<n<<",\"partition_free_energy_w3\":"<<a.w3[0]<<",\"partition_free_energy_w5\":"<<a.w5[n]<<",\"row_probability_mass_max\":"<<massmax<<",\"pair_probability_max\":"<<pmax<<",\"state_values\":"<<b.size()<<"}";std::cout<<"case "<<q<<" n "<<n<<" values "<<b.size()<<" freeenergy "<<a.w3[0]<<" rowmass "<<massmax<<"\n";
}
summary<<"],\"CPU_final_state_values\":"<<gold.size()<<",\"small_case_trace_values\":"<<trace.size()<<"}\n";save("sequences_i32.bin",seqs);save("base_can_pair_i32.bin",pairs);save("CPU_final_states_f32.bin",gold);save("CPU_trace27_allstages_f32.bin",trace);return0: return 0;}
