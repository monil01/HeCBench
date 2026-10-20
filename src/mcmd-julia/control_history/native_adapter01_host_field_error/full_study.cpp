#include <cuda_runtime.h>
#include <filesystem>
#include <iomanip>
#include <cstring>
#include "force_interface.h"
#include "force_oracle.h"
#include "configured_main.cpp"
MCMDCallback mcmd_callback=nullptr;double mcmd_excluded_seconds=0;
static std::vector<double> observed,first_input;static std::vector<std::vector<double>> rigid_reference;
static int force_calls=0,state_calls=0;static bool scientific_ok=true;static double max_force_abs=0,max_force_budget=0,max_rigid_abs=0,max_force_balance=0;
static void append(const double*p,size_t n){observed.insert(observed.end(),p,p+n);}
static void value(double x){if(!std::isfinite(x))scientific_ok=false;observed.push_back(x);}
void mcmd_capture_input(const void*p,int n,int block,int pf,double cut,double alpha,int km,int ks,double damp,const double*b,const double*rb){
 std::vector<double> x{double(n),double(block),double(pf),cut,alpha,double(km),double(ks),damp};x.insert(x.end(),b,b+9);x.insert(x.end(),rb,rb+9);const double*raw=(const double*)p;x.insert(x.end(),raw,raw+14*n);
 if(!force_calls)first_input=x;value(-16001);append(x.data(),x.size());force_calls++;
}
void mcmd_capture_output(const void*ib,const void*ob,int n,int pf,double cut,double alpha,int km,int ks,const double*b,const double*rb){
 const MAtom*a=(const MAtom*)ib,*got=(const MAtom*)ob;auto expected=independent_forces(a,n,pf,cut,alpha,km,ks,b,rb);long double sums[3]={0,0,0},magnitude=0;
 for(int i=0;i<n;i++){
  for(int byte=0;byte<112;byte++)if(!(byte>=48&&byte<72)&&((const unsigned char*)&a[i])[byte]!=((const unsigned char*)&got[i])[byte])scientific_ok=false;
  for(int c=0;c<3;c++){long double e=fabsl((long double)got[i].f[c]-expected[i][c]);double budget=(double)(e/(2e-6L+2e-10L*fabsl(expected[i][c])));max_force_abs=std::max(max_force_abs,(double)e);max_force_budget=std::max(max_force_budget,budget);if(!std::isfinite(got[i].f[c])||budget>1)scientific_ok=false;sums[c]+=got[i].f[c];magnitude+=fabsl(got[i].f[c]);}
 }
 long double balance=std::max({fabsl(sums[0]),fabsl(sums[1]),fabsl(sums[2])})/std::max(1.L,magnitude);max_force_balance=std::max(max_force_balance,(double)balance);if(balance>1e-12)scientific_ok=false;
 append((const double*)ob,14*n);
}
void mcmd_observe_system(System&s){
 auto start=std::chrono::steady_clock::now();value(-17001);value(s.stats.MDstep);value(s.stats.MDtime);value(s.molecules.size());value(s.constants.total_atoms);value(s.constants.lagrange_multiplier);value(s.constants.NH_Q);
 for(auto*m:{&s.stats.potential,&s.stats.kinetic,&s.stats.totalE,&s.stats.temp})value(m->value);
 if(!state_calls)rigid_reference.resize(s.molecules.size());
 int index=0;for(auto&m:s.molecules){
  value(m.atoms.size());value(m.mass);value(m.inertia);value(m.frozen);value(m.dof);
  for(auto*p:{m.force,m.torque,m.com,m.original_com,m.diffusion_corr,m.original_vel,m.vel,m.ov,m.ang_vel,m.ang_acc,m.old_ang_acc,m.ang_pos})for(int c=0;c<3;c++)value(p[c]);
  for(auto&a:m.atoms){value(a.mass);value(a.eps);value(a.sig);value(a.C);value(a.polar);value(a.frozen);for(auto*p:{a.pos,a.force,a.vel,a.ov,a.original_vel,a.torque,a.dip})for(int c=0;c<3;c++)value(p[c]);}
  size_t pair=0;for(size_t i=0;i<m.atoms.size();i++)for(size_t j=i+1;j<m.atoms.size();j++){double r2=0;for(int c=0;c<3;c++){double d=m.atoms[i].pos[c]-m.atoms[j].pos[c];r2+=d*d;}if(!state_calls)rigid_reference[index].push_back(r2);else{double err=std::abs(r2-rigid_reference[index][pair]);max_rigid_abs=std::max(max_rigid_abs,err);if(err>1e-7+1e-10*rigid_reference[index][pair])scientific_ok=false;}pair++;}index++;
 }state_calls++;mcmd_excluded_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
extern "C" int mcmd_run(const char*parent,MCMDCallback cb){
 try{
  mcmd_callback=cb;std::filesystem::copy_file(std::string(parent)+"/reference_history/dataset/input.pqr","input.pqr",std::filesystem::copy_options::overwrite_existing);std::filesystem::copy_file(std::string(parent)+"/reference_history/dataset/mcmd.inp","mcmd.inp",std::filesystem::copy_options::overwrite_existing);
  for(int trial=0;trial<=5;trial++){
   observed.clear();first_input.clear();rigid_reference.clear();mcmd_excluded_seconds=0;force_calls=state_calls=0;scientific_ok=true;max_force_abs=max_force_budget=max_rigid_abs=max_force_balance=0;
   char program[]="mcmd",input[]="mcmd.inp";char*argv[]={program,input,nullptr};cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();int rc=mcmd_source_main(2,argv);cudaDeviceSynchronize();double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()-mcmd_excluded_seconds;
   if(rc||force_calls!=11||state_calls!=11||first_input.size()!=47332)scientific_ok=false;
   if(!trial){printf("STUDY_WARMUP correct=%s force_calls=%d state_calls=%d oracle_max_budget=%.17g\n",scientific_ok?"true":"false",force_calls,state_calls,max_force_budget);if(!scientific_ok)return 2;continue;}
   std::vector<double> all=first_input;all.insert(all.end(),observed.begin(),observed.end());std::string path="output_trial"+std::to_string(trial)+".bin";FILE*f=fopen(path.c_str(),"wb");if(!f)return 3;fwrite(all.data(),sizeof(double),all.size(),f);fclose(f);
   printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":%zu,\"max_abs_error\":%.17g,\"max_rel_error\":null,\"max_normalized_force_budget\":%.17g,\"max_rigid_squared_distance_error\":%.17g,\"relative_total_force_balance\":%.17g,\"force_calls\":%d,\"state_calls\":%d,\"timing_region\":\"full_original_MD_CLI_pipeline_minus_study_observations_oracle\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float64\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,seconds,scientific_ok?"true":"false",all.size(),max_force_abs,max_force_budget,max_rigid_abs,max_force_balance,force_calls,state_calls,path.c_str(),all.size());fflush(stdout);
  }return 0;
 }catch(const std::exception&e){fprintf(stderr,"MCMD_SOURCE_STUDY_REJECTED %s\n",e.what());return 4;}
}
#ifndef MCMD_SHARED
int main(int argc,char**argv){return argc==2?mcmd_run(argv[1],nullptr):5;}
#endif
