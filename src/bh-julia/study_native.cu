#define main original_bh_main
#include "../bh-cuda/main.cu"
#undef main
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <set>
#include <functional>
#define CK(x) do{cudaError_t e=(x);if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}while(0)
struct Case {
 int n,steps,nn; std::vector<float>input;
 float4*p,*acc;float2*vel;float3*maxs,*mins;float*radius;
 int*sort,*child,*count,*start,*bottom,*step;unsigned int*bc;
 struct Guard{void*raw;size_t elem,len;};std::vector<Guard>guards;
 template<class T>void alloc(T*&a,int len){T*raw;CK(cudaMalloc(&raw,sizeof(T)*(len+64)));CK(cudaMemset(raw,0x5a,sizeof(T)*(len+64)));a=raw+32;CK(cudaMemset(a,0,sizeof(T)*len));guards.push_back({raw,sizeof(T),size_t(len)});}
 bool audit(int trial){bool ok=true;for(auto g:guards){std::vector<unsigned char>left(32*g.elem),right(32*g.elem);CK(cudaMemcpy(left.data(),g.raw,left.size(),cudaMemcpyDeviceToHost));CK(cudaMemcpy(right.data(),(char*)g.raw+(g.len+32)*g.elem,right.size(),cudaMemcpyDeviceToHost));for(auto x:left)ok&=x==0x5a;for(auto x:right)ok&=x==0x5a;}
 int lo;CK(cudaMemcpy(&lo,bottom,4,cudaMemcpyDeviceToHost));if(lo<n||lo>nn){printf("STUDY_TREE_CHECK invalid_bottom=%d\n",lo);return false;}
 std::vector<int>ch(8*(nn+1)),ct(nn+1),st(nn+1),order(n);std::vector<float4>pp(nn+1);CK(cudaMemcpy(ch.data(),child,ch.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(ct.data(),count,ct.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(st.data(),start,st.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(order.data(),sort,n*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(pp.data(),p,pp.size()*sizeof(float4),cudaMemcpyDeviceToHost));std::set<int>nodes,leaves;std::vector<int>dfs;int maxdepth=0;bool topology=true;
 std::function<std::pair<int,double>(int,int)>visit=[&](int node,int depth)->std::pair<int,double>{if(depth>31){topology=false;return{0,0};}if(node<n){if(node<0||!leaves.insert(node).second){topology=false;return{0,0};}dfs.push_back(node);return{1,pp[node].w};}if(node<lo||node>nn||!nodes.insert(node).second){topology=false;return{0,0};}maxdepth=std::max(maxdepth,depth);int first=dfs.size(),sum=0;double mass=0;bool empty=false;for(int j=0;j<8;j++){int c=ch[8*node+j];if(c==-1)empty=true;else if(c<0||empty){topology=false;return{0,0};}else{auto child=visit(c,depth+1);sum+=child.first;mass+=child.second;if(!topology)return{0,0};}}topology&=sum>0&&ct[node]==sum&&st[node]==first&&std::isfinite(pp[node].x)&&std::isfinite(pp[node].y)&&std::isfinite(pp[node].z)&&std::isfinite(pp[node].w)&&fabs(double(pp[node].w)-mass)<=1e-6;return{sum,mass};};auto root=visit(nn,0);topology&=root.first==n&&leaves.size()==size_t(n)&&nodes.size()==size_t(nn-lo+1)&&dfs==order&&fabs(root.second-1)<=1e-6;printf("STUDY_TREE_CHECK trial=%d n=%d topology=%s nodes=%zu maxdepth=%d guards=%s allocations=%zu\n",trial,n,topology?"true":"false",nodes.size(),maxdepth,ok?"true":"false",guards.size());return topology&&ok;
 }
 Case(int N,int S):n(N),steps(S){nn=std::max(2*n,24576);while(nn%32)nn++;nn--;
  input.resize(10*n);std::ifstream f("fixture_"+std::to_string(n)+".bin",std::ios::binary);f.read((char*)input.data(),input.size()*4);if(!f)throw std::runtime_error("fixture");
  alloc(p,nn+1);alloc(acc,nn+1);alloc(vel,nn+1);alloc(maxs,48);alloc(mins,48);alloc(radius,1);alloc(sort,nn+1);alloc(child,(nn+1)*8);alloc(count,nn+1);alloc(start,nn+1);alloc(bottom,1);alloc(step,1);alloc(bc,1);reset();}
 void reset(){CK(cudaMemcpy(p,input.data(),4*n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(vel,input.data()+4*n,2*n*4,cudaMemcpyHostToDevice));CK(cudaMemcpy(acc,input.data()+6*n,4*n*4,cudaMemcpyHostToDevice));}
 void run(){InitializationKernel<<<1,1>>>(step,bc);for(int s=0;s<steps;s++){
  BoundingBoxKernel<<<48,256>>>(nn,n,start,child,p,maxs,mins,radius,bottom,step,bc);
  ClearKernel1<<<24,256>>>(nn,n,child);
  TreeBuildingKernel<<<48,256>>>(nn,n,child,p,radius,bottom);
  ClearKernel2<<<24,256>>>(nn,start,p,bottom);
  SummarizationKernel<<<24,256>>>(nn,n,count,child,p,bottom);
  SortKernel<<<24,256>>>(nn,n,sort,count,start,child,bottom);
  ForceCalculationKernel<<<48,256>>>(nn,n,.0125f,4.f,.0025f,sort,child,p,vel,acc,radius,step);
  IntegrationKernel<<<48,256>>>(n,.025f,.0125f,p,vel,acc);
 }CK(cudaGetLastError());CK(cudaDeviceSynchronize());}
 bool capture(std::vector<float>&out,int trial){std::vector<float>x(10*n);CK(cudaMemcpy(x.data(),p,4*n*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(x.data()+4*n,vel,2*n*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(x.data()+6*n,acc,4*n*4,cudaMemcpyDeviceToHost));out.insert(out.end(),x.begin(),x.end());
  std::vector<int>order(n);CK(cudaMemcpy(order.data(),sort,n*4,cudaMemcpyDeviceToHost));std::sort(order.begin(),order.end());for(int i=0;i<n;i++)if(order[i]!=i)throw std::runtime_error("permutation");int ct;CK(cudaMemcpy(&ct,count+nn,4,cudaMemcpyDeviceToHost));if(ct!=n)throw std::runtime_error("rootcount");return audit(trial);
 }
};
int main(){std::vector<Case*>cases;for(auto c:std::vector<std::pair<int,int>>{{256,1},{256,3},{1024,1},{1024,3},{2048,1}})cases.push_back(new Case(c.first,c.second));
 std::ifstream f("expected_f64.bin",std::ios::binary|std::ios::ate);std::vector<double>expected(f.tellg()/8);f.seekg(0);f.read((char*)expected.data(),expected.size()*8);
 for(auto c:cases)c->run();bool all=true;
 for(int trial=1;trial<=5;trial++){
  std::vector<float>out;for(auto c:cases)out.insert(out.end(),c->input.begin(),c->input.end());std::vector<double>times;bool treeok=true;
  for(auto c:cases){c->reset();CK(cudaDeviceSynchronize());auto t=std::chrono::steady_clock::now();c->run();times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count());treeok&=c->capture(out,trial);}
  size_t prefix=0;for(auto c:cases)prefix+=c->input.size();bool ok=treeok;double err=0,rel=0;size_t bad=0;
  for(size_t i=0;i<expected.size();i++){double a=out[prefix+i],b=expected[i],e=fabs(a-b);err=std::max(err,e);rel=std::max(rel,e/std::max(fabs(b),1e-30));if(!std::isfinite(a)||e>1e-5+1e-4*fabs(b)){ok=false;bad++;}}
  all&=ok;std::string path=std::string(getenv("STUDY_OUTPUT_DIR")?getenv("STUDY_OUTPUT_DIR"):".")+"/bh_trial"+std::to_string(trial)+".bin";std::ofstream fo(path,std::ios::binary);fo.write((char*)out.data(),out.size()*4);fo.close();double total=0;std::string comps;for(double x:times){total+=x;if(comps.size())comps+=",";comps+=std::to_string(x);}
  printf("STUDY_TRIAL {\"trial\":%d,\"time_seconds\":%.12g,\"correct\":%s,\"checked_values\":%zu,\"mismatches\":%zu,\"max_abs_error\":%.12g,\"max_rel_error\":%.12g,\"jit_excluded\":true,\"component_seconds\":[%s],\"output_artifact\":{\"path\":\"%s\",\"dtype\":\"float32\",\"count\":%zu,\"byte_order\":\"little\"}}\n",trial,total,ok?"true":"false",expected.size(),bad,err,rel,comps.c_str(),path.c_str(),out.size());
 }
 puts(all?"PASS":"FAIL");return all?0:2;}
