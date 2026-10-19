#pragma once
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <set>
#ifndef GRAPH_ROOT
#define GRAPH_ROOT "/home/imo/HeCBench_julia/src/graphB+-julia"
#endif
using V=std::vector<int64_t>;
struct Oracle{int depth;V parent,subtree,potential,labels,count;uint64_t hi;std::vector<std::pair<int,int>>pairs;V minus,tree,inCC,inTree,negCnt;};
struct Case{int n,e,it;V idx,list,weight,orig,roots;std::vector<Oracle>oracle;};
inline std::vector<Case>load(){std::ifstream f(std::string(GRAPH_ROOT)+"/fixtures.bin",std::ios::binary);if(!f)throw std::runtime_error("fixtures missing");V a;int64_t x;while(f.read((char*)&x,8))a.push_back(x);size_t p=0;auto get=[&](size_t n){if(p+n>a.size())throw std::runtime_error("fixture bounds");V v(a.begin()+p,a.begin()+p+n);p+=n;return v;};int nc=a.at(p++);std::vector<Case>cs;for(int k=0;k<nc;k++){Case c;c.n=a.at(p++);c.e=a.at(p++);c.it=a.at(p++);c.idx=get(c.n+1);c.list=get(c.e);c.weight=get(c.e);c.orig=get(c.n);c.roots=get(c.it);for(int it=0;it<c.it;it++){Oracle o;o.depth=a.at(p++);o.parent=get(c.n);o.subtree=get(c.n);o.potential=get(c.n);o.labels=get(c.n);o.count=get(c.n);o.hi=a.at(p++);int w=a.at(p++);for(int j=0;j<w;j++){int u=a.at(p++),v=a.at(p++);o.pairs.emplace_back(u,v);}o.minus=get(c.e);o.tree=get(c.e);o.inCC=get(c.n);o.inTree=get(c.e);o.negCnt=get(c.e);c.oracle.push_back(o);}cs.push_back(c);}if(p!=a.size())throw std::runtime_error("fixture trailing values");return cs;}
inline void app(V&out,const V&a){out.insert(out.end(),a.begin(),a.end());}
// Raw snapshot node order: parent, DFS labels (before union), CC labels, distance/count, inCC;
// canonical edge rows: encoded neighbor, beg,end,inTree,negCnt,minus. Every directed edge retained.
inline bool check_snapshot(const Case&c,int it,const V&s){auto&o=c.oracle[it];size_t p=0;auto get=[&](size_t n){if(p+n>s.size())throw std::runtime_error("snapshot bounds");V v(s.begin()+p,s.begin()+p+n);p+=n;return v;};auto parent=get(c.n),dfs=get(c.n),labels=get(c.n),count=get(c.n),inCC=get(c.n);bool ok=true;auto eq=[&](const V&a,const V&b,const char*name){if(a!=b){size_t i=0;while(i<a.size()&&i<b.size()&&a[i]==b[i])i++;fprintf(stderr,"graph n%d it%d %s mismatch at%zu got%lld expected%lld\n",c.n,it,name,i,i<a.size()?(long long)a[i]:-9,i<b.size()?(long long)b[i]:-9);ok=false;}};eq(parent,o.parent,"parent");eq(labels,o.labels,"CC");eq(count,o.count,"distance");eq(inCC,o.inCC,"inCC");std::set<int64_t>seen(dfs.begin(),dfs.end());if(seen.size()!=size_t(c.n)||*seen.begin()!=0||*seen.rbegin()!=2*(c.n-1))ok=false;for(auto v:dfs)if(v&1)ok=false;int root=c.roots[it];if(dfs[root]!=0)ok=false;
for(int u=0;u<c.n;u++)for(int j=c.idx[u];j<c.idx[u+1];j++){int v=c.list[j];auto row=get(6);int tree=o.tree[j];if(row[0]!=(v*2+tree)||row[3]!=o.inTree[j]||row[4]!=o.negCnt[j]||row[5]!=o.minus[j]){fprintf(stderr,"graph n%d it%d edge%d->%d got flag%lld inTree%lld neg%lld minus%lld expected %d %lld %lld %lld\n",c.n,it,u,v,(long long)row[0],(long long)row[3],(long long)row[4],(long long)row[5],v*2+tree,(long long)o.inTree[j],(long long)o.negCnt[j],(long long)o.minus[j]);ok=false;}if((row[2]&1)!=(c.weight[j]==-1))ok=false;if(tree){bool child=(v!=root&&(parent[v]>>2)==u);int64_t beg=child?dfs[v]:(dfs[u]|1),end=((child?dfs[v]:dfs[u])+2*(o.subtree[child?v:u]-1))|int(c.weight[j]==-1);if(row[1]!=beg||row[2]!=end){fprintf(stderr,"tree interval n%d %d->%d got%lld/%lld expected%lld/%lld\n",c.n,u,v,(long long)row[1],(long long)row[2],(long long)beg,(long long)end);ok=false;}}}
if(uint64_t(s.at(p++))!=o.hi)ok=false;int w=s.at(p++);std::vector<std::pair<int,int>>pairs;for(int j=0;j<w;j++){int u=s.at(p++),v=s.at(p++);pairs.emplace_back(u,v);}if(pairs!=o.pairs)ok=false;int depth=s.at(p++);if(depth!=o.depth)ok=false;int guards=s.at(p++);if(guards!=0)ok=false;if(p!=s.size())ok=false;return ok;}
inline bool check_record(const int64_t*a,size_t len){try{auto cs=load();size_t p=0;bool ok=true;for(auto&c:cs){V expected;app(expected,c.idx);for(auto v:c.list)expected.push_back(v*2);app(expected,c.weight);app(expected,c.orig);app(expected,c.roots);for(auto v:expected)if(p>=len||a[p++]!=v)ok=false;}for(auto&c:cs){int nt=a[p++];if(nt!=c.it+1)return false;V final;for(int k=0;k<nt;k++){int n=a[p++];if(n<0||p+n>len)return false;V snap(a+p,a+p+n);p+=n;ok=check_snapshot(c,k==0?c.it-1:k-1,snap)&&ok;if(k==0)final=snap;if(k==nt-1&&snap!=final){fprintf(stderr,"timed vs replay final mismatch\n");ok=false;}}}return ok&&p==len;}catch(const std::exception&e){fprintf(stderr,"check exception %s\n",e.what());return false;}}
