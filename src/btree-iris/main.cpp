#include <iris/iris.hpp>
#include <algorithm>
#include <random>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
int main(int argc,char**argv){
 uint32_t n=1<<20,q=n;
 if(argc>1)n=std::atoi(argv[1]);q=n;
 if(argc>2)q=std::atoi(argv[2]);
 if(!n||!q)return 2;
 std::vector<uint32_t> keys(n),values(n),queries(size_t(q)*2),results(q);
 std::random_device rd;std::mt19937 g(rd());
 for(uint32_t i=0;i<n;i++)keys[i]=i;
 std::shuffle(keys.begin(),keys.end(),g);values=keys;
 for(size_t i=0;i<queries.size();i++)queries[i]=i;
 std::shuffle(queries.begin(),queries.end(),g);
 printf("Building the tree with %u keys\nSearching the tree with %u queries\n",n,q);
 constexpr size_t capacity=size_t(1)<<30;
 auto *pool=(uint32_t*)malloc(capacity*sizeof(uint32_t));
 if(!pool)return 2;
 uint32_t root=0,count=0;
 iris::Platform platform;platform.init(&argc,&argv,1);
 int devices=0;iris_device_count(&devices);
 for(int d=0;d<devices;d++){int backend=0;size_t bytes=0;char name[256]={};iris_device_info(d,iris_backend,&backend,&bytes);iris_device_info(d,iris_name,name,&bytes);printf("IRIS device %d: %s backend=%d (HIP=%d)\n",d,name,backend,iris_hip);}
 {
 iris::DMem mp(pool,capacity*4),mr(&root,4),mc(&count,4),mk(keys.data(),size_t(n)*4),mv(values.data(),size_t(n)*4),mq(queries.data(),size_t(q)*4),mo(results.data(),size_t(q)*4);
 auto start=std::chrono::steady_clock::now();
 size_t global=capacity,local=256;
 iris::Task clear;void* cp[]={&mp,(void*)&capacity};int ci[]={iris_w,sizeof(capacity)};
 clear.kernel("clear_pool",1,nullptr,&global,&local,2,cp,ci);clear.submit(iris_gpu,nullptr,1);
 global=local=32;
 iris::Task init;void* ip[]={&mp,&mc,&mr};int ii[]={iris_rw,iris_w,iris_w};
 init.kernel("init_tree",1,nullptr,&global,&local,3,ip,ii);init.submit(iris_gpu,nullptr,1);
 global=(size_t(n)+31)/32*32;
 iris::Task build;void* bp[]={&mr,&mk,&mv,&n,&mp,&mc};int bi[]={iris_r,iris_r,iris_r,sizeof(n),iris_rw,iris_rw};
 build.kernel("insert_keys",1,nullptr,&global,&local,6,bp,bi);build.submit(iris_gpu,nullptr,1);
 global=(size_t(q)+31)/32*32;
 iris::Task search;void* sp[]={&mr,&mq,&mo,&q,&mp,&mc};int si[]={iris_r,iris_r,iris_w,sizeof(q),iris_r,iris_r};
 search.kernel("search_keys",1,nullptr,&global,&local,6,sp,si);search.flush_out(mo);search.flush_out(mc);search.submit(iris_gpu,nullptr,1);
 printf("Total device execution time = %.3f s\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
 }
 platform.finalize();free(pool);
 size_t exist=0,errors=0;
 for(uint32_t i=0;i<q;i++){uint32_t expected=queries[i]<n?queries[i]:0;exist+=queries[i]<n;errors+=results[i]!=expected;}
 printf("numeric_check queries=%u mismatches=%zu\n",q,errors);
 if(!errors)printf("PASS. ([%0.2f%%] queries exist in search.)\n",100.*exist/q);
 printf("Tree size: %f GiBs.\n",double(count)*32*4/(1u<<30));
 return errors?1:0;
}
