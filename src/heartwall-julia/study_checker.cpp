#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <cmath>
#include <cstring>
#include "study_params.h"
#include "native_field_offsets.h"
template<class T>std::vector<T>read(const std::string&p,size_t count=0){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)exit(2);if(!count)count=f.tellg()/sizeof(T);f.seekg(0);std::vector<T>x(count);f.read((char*)x.data(),count*sizeof(T));if(size_t(f.gcount())!=count*sizeof(T))exit(3);return x;}
struct Field{int n;};
extern "C" int check_heartwall(const float*ptr,int count,int frame,int last,double*out_error){
static auto expected=read<double>("oracle_final_f64.bin",FINAL_COUNT);static auto budget=read<double>("oracle_final_budget_f64.bin",FINAL_COUNT);static auto expectedNCC=read<double>("oracle_NCC_f64.bin");static auto expectedCov=read<double>("oracle_covariance_f64.bin");static auto expectedCovBudget=read<double>("oracle_covariance_budget_f64.bin");static auto expectedBox=read<double>("oracle_boxsum_f64.bin");static auto expectedR=read<int>("oracle_track_rows_i32.bin");static auto expectedC=read<int>("oracle_track_cols_i32.bin");static auto initialT=read<float>("oracle_template_initial_f32.bin");std::vector<std::unique_ptr<Field>>f;
f.emplace_back(new Field{20});
f.emplace_back(new Field{20});
f.emplace_back(new Field{2080});
f.emplace_back(new Field{2080});
f.emplace_back(new Field{31});
f.emplace_back(new Field{31});
f.emplace_back(new Field{3224});
f.emplace_back(new Field{3224});
f.emplace_back(new Field{52020});
f.emplace_back(new Field{80631});
f.emplace_back(new Field{334611});
f.emplace_back(new Field{875211});
f.emplace_back(new Field{1707939});
f.emplace_back(new Field{1222623});
f.emplace_back(new Field{1222623});
f.emplace_back(new Field{875211});
f.emplace_back(new Field{875211});
f.emplace_back(new Field{334611});
f.emplace_back(new Field{875211});
f.emplace_back(new Field{132651});
f.emplace_back(new Field{875211});
f.emplace_back(new Field{875211});
f.emplace_back(new Field{132651});
f.emplace_back(new Field{2601});
f.emplace_back(new Field{2601});
f.emplace_back(new Field{6681});
f.emplace_back(new Field{6681});
f.emplace_back(new Field{51});
f.emplace_back(new Field{51});
f.emplace_back(new Field{51});
auto check=[&](const std::vector<float>&x,int frame,bool last,double&maxerr){bool ok=x.size()==FINAL_COUNT;for(size_t i=0;i<x.size();i++)ok=ok&&std::isfinite(x[i]);auto get=[&](int off){return x.data()+off+16;};if(last){for(size_t i=0;i<x.size();i++)if(budget[i]>=0){double e=fabs(double(x[i])-expected[i]);maxerr=fmax(maxerr,e);if(e>budget[i]){if(ok)fprintf(stderr,"oraclefinal failure index%zu actual%.17g ref%.17g error%.17g budget%.17g\n",i,double(x[i]),expected[i],e,budget[i]);ok=false;}}}else{for(size_t k=0;k<f.size();k++){// guard equality is checked in every actual frame, including unused-zero stage0.
 size_t off=0;for(size_t j=0;j<k;j++)off+=f[j]->n+32;for(int j=0;j<16;j++)ok=ok&&x[off+j]==expected[off+j]&&x[off+f[k]->n+16+j]==expected[off+f[k]->n+16+j];}}
const float*er=get(OFF_tEndoRowLoc),*ec=get(OFF_tEndoColLoc),*pr=get(OFF_tEpiRowLoc),*pc=get(OFF_tEpiColLoc);for(int p=0;p<51;p++)for(int t=0;t<104;t++){float rr=t<=frame?expectedR[t*51+p]:0,cc=t<=frame?expectedC[t*51+p]:0;ok=ok&&(p<20?er[p*104+t]:pr[(p-20)*104+t])==rr&&(p<20?ec[p*104+t]:pc[(p-20)*104+t])==cc;}if(frame<10){const float*et=get(OFF_endoT),*pt=get(OFF_epiT);for(int p=0;p<51;p++)for(int k=0;k<2601;k++)ok=ok&&(p<20?et[p*2601+k]:pt[(p-20)*2601+k])==initialT[p*2601+k];}
if(frame>0){const float*ncc=get(OFF_in2_sqr_sub2),*cov=get(OFF_conv),*box=get(OFF_in2_sub2),*mask=get(OFF_mask_conv),*pv=get(OFF_par_max_val),*pi=get(OFF_par_max_coo);size_t offset=size_t(frame-1)*875211;for(int i=0;i<875211;i++){ok=ok&&fabs(double(ncc[i])-expectedNCC[offset+i])<=1e-3&&fabs(double(cov[i])-expectedCov[offset+i])<=expectedCovBudget[offset+i]&&double(box[i])==expectedBox[offset+i];}for(int p=0;p<51;p++)for(int col=0;col<131;col++){float maximum=0;int index=0;for(int row=0;row<131;row++){int k=col*131+row;float value=fabsf(mask[p*17161+k]);if(value>maximum){maximum=value;index=k;}}ok=ok&&pv[p*131+col]==maximum&&pi[p*131+col]==float(index);}}
return ok;};
std::vector<float>x(ptr,ptr+count); double error=0;bool good=check(x,frame,last,error);*out_error=error;return good?1:0;
}
