// Source CPU API vs the literal CUDA force equation; not GPU execution evidence.
#include <cstdio>
#include <cmath>
#include <initializer_list>
#include "reference_history/ForceLaw.cxx"
int main(){FGrid grid;FGridEvalFit fit(&grid);ForceLawNewton newton;ForceLawSR sr(&fit,.1f);
 printf("{\"source_conventions\":[");bool first=true;for(float r:{.25f,.5f,1.f}){
  float r2=r*r;double gpu_rule=1/std::pow(double(r2)+double(.1f*.1f),1.5);
  printf("%s{\"r\":%.9g,\"r2\":%.9g,\"CPU_Newton_f_over_r\":%.17g,\"CPU_SR_f_over_r\":%.17g,\"CUDA_literal_softened_cutoff_rule_F64\":%.17g,\"Newton_relative_discrepancy\":%.17g,\"source_removed_grid_term\":%.17g}",first?"":",",r,r2,double(newton.f_over_r(r2)),double(sr.f_over_r(r2)),gpu_rule,std::abs(gpu_rule-newton.f_over_r(r2))/std::abs(newton.f_over_r(r2)),double(fit.eval(r2)));first=false;
 }puts("],\"domain\":\"positive distances below3.116326355 sourcecutoff,unitmass,smoothing.1\",\"not_GPU_output_evidence\":true}");}
