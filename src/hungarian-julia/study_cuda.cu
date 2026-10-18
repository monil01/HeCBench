#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <climits>
#define _n_ 256
#define _range_ 256
#define main original_hungarian_main
#include "../hungarian-cuda/main.cu"
#undef main
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(e));exit(10);}}while(0)
#define RUN(k,b,t,s,id) do{k<<<b,t,s>>>();CK(cudaDeviceSynchronize());counts[id]++;if(++launches>10000){fprintf(stderr,"iterationcap\n");exit(11);}}while(0)
std::vector<int>solve(){std::vector<int>counts(16);int launches=0;
RUN(init,n_blocks,n_threads,0,0);RUN(calc_min_in_rows,n_blocks_reduction,n_threads_reduction,0,1);RUN(step_1_row_sub,n_blocks_full,n_threads_full,0,2);RUN(calc_min_in_cols,n_blocks_reduction,n_threads_reduction,0,3);RUN(step_1_col_sub,n_blocks_full,n_threads_full,0,4);RUN(compress_matrix,n_blocks_full,n_threads_full,0,5);
do{repeat_kernel=false;RUN(step_2,1,zeros_size>256?256:zeros_size,0,6);}while(repeat_kernel);
while(1){RUN(step_3ini,n_blocks,n_threads,0,7);RUN(step_3,n_blocks,n_threads,0,8);if(n_matches>=n)break;RUN(step_4_init,n_blocks,n_threads,0,9);
while(1){do{goto_5=false;repeat_kernel=false;RUN(step_4,1,zeros_size>256?256:zeros_size,0,10);}while(repeat_kernel&&!goto_5);if(goto_5)break;
RUN(min_reduce_kernel1,n_blocks_reduction,n_threads_reduction,n_threads_reduction*4,13);RUN(min_reduce_kernel2,1,n_blocks_reduction/2,n_blocks_reduction/2*4,14);RUN(step_6_add_sub,n_blocks_full,n_threads_full,0,15);RUN(compress_matrix,n_blocks_full,n_threads_full,0,5);}
RUN(step_5a,n_blocks,n_threads,0,11);RUN(step_5b,n_blocks,n_threads,0,12);}return counts;}
std::vector<int>load(const char*path,int count){std::vector<int>x(count);FILE*f=fopen(path,"rb");if(!f||fread(x.data(),4,count,f)!=count)exit(3);fclose(f);return x;}
int main(){static_assert(sizeof(int)==4);auto prefix=load("/home/imo/HeCBench_julia/src/hungarian-julia/fixture.bin",196610),gold=load("/home/imo/HeCBench_julia/src/hungarian-julia/oracle.bin",198147);auto reset=[&](int ci){CK(cudaMemcpyToSymbol(slack,prefix.data()+2+ci*n*n,n*n*4));};for(int ci=0;ci<3;ci++){reset(ci);solve();}
for(int trial=1;trial<=5;trial++){bool ok=true;std::vector<int>art=prefix,counts,raw;std::vector<double>paths;double elapsed=0;int go=0;
for(int ci=0;ci<3;ci++){reset(ci);CK(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();auto c=solve();CK(cudaDeviceSynchronize());double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();elapsed+=dt;paths.push_back(dt);counts.insert(counts.end(),c.begin(),c.end());std::vector<int>cs(n),rs(n),cost(n*n);CK(cudaMemcpyFromSymbol(cs.data(),column_of_star_at_row,n*4));CK(cudaMemcpyFromSymbol(rs.data(),row_of_star_at_column,n*4));CK(cudaMemcpyFromSymbol(cost.data(),slack,n*n*4));raw.insert(raw.end(),cost.begin(),cost.end());raw.insert(raw.end(),cs.begin(),cs.end());raw.insert(raw.end(),rs.begin(),rs.end());long total=0;for(int r=0;r<n;r++){int col=cs[r];ok&=col>=0&&col<n;if(col>=0&&col<n){total+=prefix[2+ci*n*n+col*n+r];ok&=rs[col]==r&&cost[col*n+r]==0;}}for(int x:cost)ok&=x>=0;art.insert(art.end(),cs.begin(),cs.end());art.insert(art.end(),rs.begin(),rs.end());art.push_back(int(total));art.insert(art.end(),prefix.begin()+2+ci*n*n,prefix.begin()+2+(ci+1)*n*n);}
if(art.size()!=394757)return 4;for(int i=196610;i<394757;i++)ok&=art[i]==gold[go++];std::string base=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".");std::string file=base+"/output_trial"+std::to_string(trial)+".bin",aux=base+"/raw_slack_assignment_trial"+std::to_string(trial)+".bin";FILE*f=fopen(file.c_str(),"wb");fwrite(art.data(),4,art.size(),f);fclose(f);f=fopen(aux.c_str(),"wb");fwrite(raw.data(),4,raw.size(),f);fclose(f);
printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.17g,\"correct\":%s,\"checked_values\":198147,\"max_abs_error\":null,\"max_rel_error\":null,\"timing_region\":\"sum3_full_Hungarian_host_controlled_GPU_solvers\",\"jit_excluded\":true,\"internal_repeats\":1,\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":394757,\"byte_order\":\"little\"},\"auxiliary_artifacts\":[{\"path\":\"%s\",\"dtype\":\"int32\",\"count\":198144}],\"path_times_seconds\":[",trial,elapsed,ok?"true":"false",file.c_str(),aux.c_str());for(int i=0;i<3;i++)printf("%s%.17g",i?",":"",paths[i]);printf("],\"stage_launch_counts\":[");for(int i=0;i<48;i++)printf("%s%d",i?",":"",counts[i]);printf("]}\n");}}
