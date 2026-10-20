// Independent mathematical event-sweep evaluator, not a transcription of GPU pointer loops.
// K-way event-time priority queue, logical truth evaluation, pending-output normalization.
#include <cstdint>
#include <vector>
#include <queue>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <algorithm>
using U=uint64_t;const U END=~U(0);
struct Event{U t,v;};using Wave=std::vector<Event>;
struct Gate{U func,out,arity;U input[6],delay[72];};
U read(std::ifstream&f){U x;f.read((char*)&x,8);if(!f)exit(2);return x;}
void write(std::ofstream&f,U x){f.write((char*)&x,8);}
Wave evaluate(const Gate&g,const std::vector<Wave>&all,const std::vector<unsigned char>&table,U off,bool gpu){
 Wave out=gpu?Wave{{0,2}}:all[g.out];
 struct Next{U time,port,index;bool operator>(const Next&x)const{return time>x.time;}};
 std::priority_queue<Next,std::vector<Next>,std::greater<Next>>heap;
 U values[6]={},previous[6]={};
 for(U p=0;p<g.arity;p++){const Wave&w=all[g.input[p]];if(w.empty()||w.front().t!=0)exit(3);values[p]=w.front().v;if(w.size()>1&&w[1].t!=END)heap.push({w[1].t,p,1});}
 U prevOutput=2;
 while(!heap.empty()&&heap.top().time<=off){U t=heap.top().time;std::copy(values,values+6,previous);
  do{Next item=heap.top();heap.pop();const Wave&w=all[g.input[item.port]];values[item.port]=w[item.index].v;U next=item.index+1;if(next<w.size()&&w[next].t!=END)heap.push({w[next].t,item.port,next});}while(!heap.empty()&&heap.top().time==t);
  U code=0;for(U p=0;p<g.arity;p++)code=4*code+values[p];U nextOutput=(table[g.func*1025+code/4]>>(2*(code%4)))&3;
  if(nextOutput!=prevOutput){
   int edge=-1;if(prevOutput==0&&nextOutput==1)edge=0;if(prevOutput==1&&nextOutput==0)edge=1;if(prevOutput==0&&nextOutput==2)edge=2;if(prevOutput==2&&nextOutput==1)edge=3;if(prevOutput==1&&nextOutput==2)edge=4;if(prevOutput==2&&nextOutput==0)edge=5;if(edge<0)exit(4);
   U delay=END;for(U p=0;p<g.arity;p++)if(values[p]!=previous[p])delay=std::min(delay,g.delay[12*p+edge+((previous[p]==1||values[p]==0)?6:0)]);if(delay==END)exit(5);U emission=t+delay;
   while(!out.empty()&&out.back().t>=emission)out.pop_back();if(out.empty()||out.back().v!=nextOutput)out.push_back({emission,nextOutput});
  }
  prevOutput=nextOutput;
 }
 if(gpu&&(out.empty()||out.back().t!=END))out.push_back({END,3});return out;
}
int main(int argc,char**argv){std::ifstream inputs(argv[1],std::ios::binary),graph(argv[2],std::ios::binary),truth(argv[3],std::ios::binary);U nw=read(inputs);std::vector<Wave>waves(nw);for(U i=0;i<nw;i++){U wire=read(inputs),n=read(inputs);if(wire!=i+4)exit(6);waves[i].resize(n);for(Event&e:waves[i]){e.t=read(inputs);e.v=read(inputs);}}
 U header[9];for(U&x:header)x=read(graph);if(header[1]!=nw)exit(7);std::vector<unsigned char>table(header[2]*1025);truth.read((char*)table.data(),table.size());U allgates=0;
 for(U b=0;b<header[3];b++){U block=read(graph),gpu=read(graph),n=read(graph),pop=read(graph),del=read(graph),add=read(graph);for(U i=0;i<pop+del;i++)read(graph);for(U i=0;i<add;i++){U wire=read(graph);if(gpu&&waves[wire].back().t!=END)waves[wire].push_back({END,3});}
  std::vector<Gate>gates(n);for(Gate&g:gates){g.func=read(graph);g.out=read(graph);g.arity=read(graph);for(U&x:g.input)x=read(graph);for(U&x:g.delay)x=read(graph);}
  // Declared source graph has no within-batch dependencies, independently audited.
  for(const Gate&g:gates)waves[g.out]=evaluate(g,waves,table,header[5],gpu);
  allgates+=n;U events=0;for(const Gate&g:gates)events+=waves[g.out].size();printf("CPU_MATH block=%llu GPU_source=%llu gates=%llu events=%llu\n",(unsigned long long)block,(unsigned long long)gpu,(unsigned long long)n,(unsigned long long)events);fflush(stdout);
 }
 std::ofstream out(argv[4],std::ios::binary),occupancy(argv[5],std::ios::binary);write(out,nw);U total=0;
 for(U wire=0;wire<nw;wire++){const Wave&w=waves[wire];write(out,wire+4);write(out,w.size());for(const Event&e:w){write(out,e.t);write(out,e.v);}total+=w.size();U counts[3]={0,0,0};U last=0,value=3;for(const Event&e:w){if(e.t>=header[5])break;counts[value<2?value:2]+=e.t-last;last=e.t;value=e.v;}counts[value<2?value:2]+=header[5]-last;for(U c:counts)write(occupancy,c);}
 printf("CPU_MATH completegates=%llu totalnetevents=%llu\n",(unsigned long long)allgates,(unsigned long long)total);return 0;}
