// Serial source crosscheck and raw original table preparation, not an independent
// folding algorithm. Independent mathematical certificates are separate.
#define INT
#include "native_kernels.cuh"
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <array>
struct SerialState{
 int n;std::vector<fbase_t>s;std::vector<int_t>v,w,wm,wca,w5base,w3;
 SerialState(const std::string&str):n(str.size()),v(n*n,INF),w(v),wm(v),wca(v),w5base(n+1),w3(n+1){for(char c:str)s.push_back(fbase_from_char(c));init_w5_and_w3(n,w5base.data(),w3.data());}
 void step(int d,fparam_t p){auto a=s.data();auto vv=v.data();auto ww=w.data();auto mm=wm.data();auto pre=w5base.data()+1;auto suf=w3.data();auto ca=wca.data();calc_V_hairpin_and_V_stack(d,n,a,vv,p);calc_V_bulge_internal(d,n,a,vv,p);calc_V_exterior(d,n,a,vv,pre,suf,p);calc_V_multibranch(d,n,a,vv,mm,p);calc_coaxial(d,n,a,vv,ww,pre,suf,p);calc_W(d,n,a,vv,ww,p);calc_WM(d,n,a,ww,mm,p);calc_wl_coax(d,n,a,vv,ww,mm,p,ca);calc_w5_and_w3(d,n,a,vv,pre,suf,p,ca);}
 void save(std::ofstream&f){for(auto*a:{&v,&w,&wm,&wca,&w5base,&w3})f.write((char*)a->data(),a->size()*4);assert(f);}
};
int main(int argc,char**argv){assert(argc==4);fparam par{};fparam_read_from_text(argv[1],0,&par);char*original=fsequence(argv[2]);std::vector<std::string>seqs={original,"AAAAAAAAAAAA","GGGAAACCC","GCGUUUCGC","CC"+std::string(40,'A')+"GG","GGG"+std::string(10,'C')};free(original);std::string dir=argv[3];std::ofstream pf(dir+"/params.bin",std::ios::binary);pf.write((char*)&par,sizeof(par));pf.close();assert(pf);std::ofstream sf(dir+"/sequences.bin",std::ios::binary);int nc=seqs.size();sf.write((char*)&nc,4);std::ofstream gold(dir+"/serial_source_crosscheck.bin",std::ios::binary);FILE*meta=fopen((dir+"/fixture_metadata.json").c_str(),"w");assert(meta);fprintf(meta,"{\"role\":\"Serialsourcecrosscheck_notindependentfoldingalgorithm\",\"parameter_bytes\":%zu,\"cases\":[",sizeof(par));size_t offset=0;
 for(int c=0;c<nc;c++){SerialState x(seqs[c]);assert(x.n>=5);sf.write((char*)&x.n,4);sf.write((char*)x.s.data(),x.s.size()*4);size_t count=4*x.n*x.n+2*(x.n+1);x.save(gold);for(int d=0;d<x.n-1;d++){x.step(d,&par);x.save(gold);}assert(x.w5base[x.n]==x.w3[0]);assert(x.w5base[0]==0&&x.w3[x.n]==0);fprintf(meta,"%s{\"sequence\":\"%s\",\"n\":%d,\"checkpoints\":%d,\"state_values_per_checkpoint\":%zu,\"crosscheck_offset_Int32\":%zu,\"MFE_scaled_Int32\":%d,\"prefix_suffix_final_equal\":true}",c?",":"",seqs[c].c_str(),x.n,x.n,count,offset,x.w3[0]);offset+=count*x.n;}
 fprintf(meta,"],\"crosscheck_total_Int32_values\":%zu,\"parameter_prelog_Float32\":%.17g,\"unused_parameter_bytes_initially_zeroed\":true}\n",offset,double(par.prelog));fclose(meta);sf.close();gold.close();assert(sf&&gold);
 FILE*layout=fopen((dir+"/parameter_layout.json").c_str(),"w");assert(layout);fprintf(layout,"{\"struct_bytes\":%zu,\"word_bytes\":4,\"base_dimension\":4,\"fields\":{",sizeof(par));bool first=true;
#define FIELD(name) fprintf(layout,"%s\""#name"\":{\"word_offset\":%zu,\"word_count\":%zu}",first?"":",",offsetof(fparam,name)/4,sizeof(par.name)/4);first=false
 FIELD(use_dna_fparams);FIELD(coaxial);FIELD(coaxstack);FIELD(stack);FIELD(tstack);FIELD(tstackcoax);FIELD(tstackh);FIELD(tstacki);FIELD(tstacki1n);FIELD(tstacki23);FIELD(tstackm);FIELD(internal_loop_initiation);FIELD(bulge_loop_initiation);FIELD(hairpin_loop_initiation);FIELD(Extrapolation_for_large_loops);FIELD(prelog);FIELD(maximum_correction);FIELD(fm_array_first_element);FIELD(a);FIELD(b);FIELD(c);FIELD(a_2c);FIELD(a_2b_2c);FIELD(terminal_AU_penalty);FIELD(bonus_for_GGG_hairpin);FIELD(c_hairpin_slope);FIELD(c_hairpin_intercept);FIELD(c_hairpin_of_3);FIELD(Bonus_for_Single_C_bulges_adjacent_to_C);FIELD(ntriloop);FIELD(triloop);FIELD(ntloop);FIELD(tloop);FIELD(nhexaloop);FIELD(hexaloop);FIELD(dangle_3p);FIELD(dangle_5p);FIELD(int11);FIELD(int22);FIELD(int21);
#undef FIELD
 fprintf(layout,"}}\n");fclose(layout);printf("FRNA_FIXTURE_CASES %d SOURCE_STATES %zu\n",nc,offset);}
