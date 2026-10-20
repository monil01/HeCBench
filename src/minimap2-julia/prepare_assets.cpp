#include "../minimap2-cuda/src/host_data_io.cpp"
#include "../minimap2-cuda/src/memory_scheduler.cpp"
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdio>
template<class T>void save(const char*name,const std::vector<T>&x){std::ofstream f(name,std::ios::binary);f.write((char*)x.data(),x.size()*sizeof(T));if(!f)throw std::runtime_error(name);}
int main(){FILE*f=fopen("study_reads.txt","r");if(!f)throw std::runtime_error("reads");std::vector<control_dt>c;std::vector<anchor_dt>a;std::vector<anchor_idx_t>ns;int dx=5000,dy=5000,bw=500;scheduler(f,c,a,ns,0x7fffffff,dx,dy,bw);fclose(f);dx=5000;dy=5000;bw=500;
// Repack source logical control fields into deterministically zeroed8-byte objects.
std::vector<uint8_t>cb(c.size()*8,0);for(size_t i=0;i<c.size();++i){memcpy(cb.data()+8*i,&c[i].avg_qspan,4);memcpy(cb.data()+8*i+4,&c[i].tile_num,2);cb[8*i+6]=c[i].is_new_read;}
save("control_u8.bin",cb);save("anchors_u16.bin",a);
size_t batches=c.size()/PE_NUM;std::vector<return_dt>r(batches*PE_NUM*TILE_SIZE);std::vector<int32_t>mx(PE_NUM*64,0),jp(PE_NUM*64,-1);
// Independent forward recurrence: finalize one anchor, propagate its score to the next64 anchors, preserve latest-predecessor tie order. No cooperative/GPU circular mutation barriers.
for(size_t batch=0;batch<batches;batch++)for(int pe=0;pe<PE_NUM;pe++){size_t ci=batch*PE_NUM+pe,ab=ci*TILE_SIZE_ACTUAL,ob=ci*TILE_SIZE;auto ct=c[ci];if(ct.is_new_read){std::fill(mx.begin()+pe*64,mx.begin()+(pe+1)*64,0);std::fill(jp.begin()+pe*64,jp.begin()+(pe+1)*64,-1);}int32_t fs[1088]={},ps[1088];std::fill(ps,ps+1088,-1);for(int k=0;k<64;k++){fs[k]=mx[pe*64+k];ps[k]=jp[pe*64+k];}for(int i=0;i<1024;i++){auto cur=a[ab+i];int score=fs[i],parent=ps[i];if(cur.w>=score){score=cur.w;parent=-1;}r[ob+i]={score,parent};for(int k=i+1;k<=i+64;k++){auto next=a[ab+k];int x=int(next.x)-int(cur.x),y=int(next.y)-int(cur.y),dd=abs(x-y);int sc=-0x3fffffff;if(cur.tag==next.tag&&x!=0&&x<=dx&&y>0&&y<=dy&&dd<=bw){int log=dd<2?0:std::min(8,31-__builtin_clz((unsigned)dd));float weight=0.01f*ct.avg_qspan;float product=float(dd)*weight;sc=std::min(std::min(x,y),int(next.w))-int(product)-(log>>1);}int candidate=sc+score;if(candidate>=fs[k]){fs[k]=candidate;ps[k]=i+int(ct.tile_num)*1024;}}}for(int k=0;k<64;k++){mx[pe*64+k]=fs[1024+k];jp[pe*64+k]=ps[1024+k];}}
save("oracle_return_i32.bin",r);save("oracle_max_i32.bin",mx);save("oracle_parent_i32.bin",jp);std::ofstream sizes("sizes.txt");sizes<<a.size()<<" "<<c.size()<<" "<<r.size()<<" "<<mx.size()<<" "<<batches<<"\n";printf("CPU source packing + independentDP: anchors%zu controls%zu returns%zu batches%zu\n",a.size(),c.size(),r.size(),batches);}
