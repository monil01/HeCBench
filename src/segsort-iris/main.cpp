/*
* (c) 2015-2019 Virginia Polytechnic Institute & State University (Virginia Tech)
*          2020 Robin Kobus (kobus@uni-mainz.de)
*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, version 2.1
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License, version 2.1, for more details.
*
*   You should have received a copy of the GNU General Public License
*
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <random>
#include <iris/iris.hpp>
#include "gpu_lock.hpp"
#include <memory>
#include <chrono>



using std::vector;
using std::pair;

using index_t = int;
using offset_t = int;
using key_t = int;
using val_t = uint64_t;





template<class K, class T>
void gold_segsort(vector<K> &key, vector<T> &val, const vector<offset_t> &seg, index_t num_segs)
{
    vector<pair<K,T>> pairs;
    for(index_t i = 0; i < num_segs; i++)
    {
        offset_t st = seg[i];
        offset_t ed = seg[i+1];
        offset_t size = ed - st;
        pairs.reserve(size);
        for(index_t j = st; j < ed; j++)
        {
            pairs.emplace_back(key[j], val[j]);
        }
        stable_sort(pairs.begin(), pairs.end(), [&](pair<K,T> a, pair<K,T> b){ return a.first < b.first;});
        // sort(pairs.begin(), pairs.begin(), [&](pair<K,T> a, pair<K,T> b){ return a.first < b.first;});

        for(index_t j = st; j < ed; j++)
        {
            key[j] = pairs[j-st].first;
            val[j] = pairs[j-st].second;
        }
        pairs.clear();
    }
}


template<class K>
void gold_segsort(vector<K> &key, const vector<offset_t> &seg, index_t num_segs)
{
    for(index_t i = 0; i < num_segs; i++)
    {
        offset_t st = seg[i];
        offset_t ed = seg[i+1];
        stable_sort(key.begin()+st, key.begin()+ed, [&](K a, K b){ return a < b;});
        // sort(key.begin()+st, key.begin()+ed, [&](K a, K b){ return a < b;});
    }
}


template<class K, class T>
void sort_vals_of_same_key(const vector<K> &key, vector<T> &val, const vector<offset_t> &seg, index_t num_segs)
{
    for(index_t i = 0; i < num_segs; i++)
    {
        offset_t st = seg[i];
        offset_t ed = seg[i+1];
        for(offset_t i = st; i < ed;)
        {
            auto range = std::equal_range(key.begin()+st, key.begin()+ed, key[i]);
            offset_t i_st = std::distance(key.begin(), range.first);
            offset_t i_ed = std::distance(key.begin(), range.second);
            sort(val.begin()+i_st, val.begin()+i_ed, [&](T a, T b){ return a < b;});
            i += i_ed - i_st;
        }
    }
}



void launch(const std::string& name,iris::DMem& key,iris::DMem& val,iris::DMem& keyB,iris::DMem& valB,iris::DMem& seg,iris::DMem& bin,int offset,int n,int stride,size_t blocks,size_t threads,size_t rows=1){
 iris::Task task;void* p[]={&key,&val,&keyB,&valB,&seg,&bin,&offset,&n,&stride};
 int info[]={iris_rw,iris_rw,iris_rw,iris_rw,iris_r,iris_r,sizeof(offset),sizeof(n),sizeof(stride)};
 size_t g[]={blocks*threads,rows},l[]={threads,1};
 task.kernel(name.c_str(),rows>1?2:1,nullptr,g,l,name.find("_long_")!=std::string::npos?9:8,p,info);
 if(task.submit(iris_gpu,nullptr,1)!=IRIS_SUCCESS)throw std::runtime_error("IRIS launch failed "+name);
}
bool segsort(int elements,bool keys_only){
 const int seed=-278642091;printf("seed: %d\n",seed);
 std::mt19937 gen(seed);std::uniform_int_distribution<int> dis(0,elements),segdis(0,5000);
 vector<int> key(elements),seg;for(auto&k:key)k=dis(gen);
 int off=0;while(off<elements){seg.push_back(off);off=seg.back()+segdis(gen);}seg.push_back(elements);
 int num=seg.size()-1;printf("synthesized segments # %d (max_size: 5000, min_size: 0)\n",num);
 vector<uint64_t> val(keys_only?1:elements);if(!keys_only)for(auto&v:val)v=dis(gen);
 auto goldkey=key;auto goldval=val;
 // Independent original CPU reference is computed outside the GPU lock.
 if(keys_only)gold_segsort(goldkey,seg,num);else gold_segsort(goldkey,goldval,seg,num);
 vector<int> output(elements),bin(num);vector<uint64_t> outval(keys_only?1:elements);
 int counter[14]={};
 {
 GpuLock lock;
 iris::DMem mk(key.data(),size_t(elements)*4),mv(val.data(),val.size()*8),mko(output.data(),size_t(elements)*4),mvo(outval.data(),outval.size()*8),ms(seg.data(),seg.size()*4),mb(bin.data(),bin.size()*4),mc(counter,sizeof(counter));
 size_t global=(size_t(num)+255)/256*256,local=256;
 iris::Task hist;void* hp[]={&mc,&ms,&num};int hi[]={iris_rw,iris_r,sizeof(num)};
 hist.kernel("bin_histo",1,nullptr,&global,&local,3,hp,hi);hist.flush_out(mc);hist.submit(iris_gpu,nullptr,1);
 int maxsize=counter[13];counter[13]=num;
 iris::Task group;void* gp[]={&mb,&mc,&ms,&num};int gi[]={iris_w,iris_rw,iris_r,sizeof(num)};
 group.kernel("bin_group",1,nullptr,&global,&local,4,gp,gi);group.submit(iris_gpu,nullptr,1);
 const size_t threads[]={256,256,128,128,128,128,128,256,256,128,256,512};
 const size_t warps[]={1,2,2,2,4,8,16,8,32,128,256,512};
 std::string mode=keys_only?"keys":"pairs";
 for(int b=0;b<12;b++){
  int count=counter[b+1]-counter[b];if(!count)continue;
  size_t factor=threads[b]/warps[b],blocks=(count+factor-1)/factor;
  launch(mode+"_bin"+std::to_string(b),mk,mv,mko,mvo,ms,mb,counter[b],count,0,blocks,threads[b]);
 }
 int count=counter[13]-counter[12];
 if(count){
  size_t rows=(maxsize+2047)/2048;
  launch(mode+"_long_sort",mk,mv,mko,mvo,ms,mb,counter[12],count,0,count,512,rows);
  iris::DMem *ki=&mko,*ko=&mk,*vi=&mvo,*vo=&mv;
  int swaps=1;
  for(int stride=2048;stride<maxsize;stride*=2){
   launch(mode+"_long_merge",*ki,*vi,*ko,*vo,ms,mb,counter[12],count,stride,count,128,rows);
   std::swap(ki,ko);std::swap(vi,vo);swaps++;
  }
  if(swaps&1){std::swap(ki,ko);std::swap(vi,vo);}
  launch(mode+"_long_copy",*ki,*vi,*ko,*vo,ms,mb,counter[12],count,0,count,128,rows);
 }
 iris::Task flush;flush.flush_out(mko);if(!keys_only)flush.flush_out(mvo);flush.submit(iris_gpu,nullptr,1);
 }
 size_t errors=0;for(int i=0;i<elements;i++)errors+=output[i]!=goldkey[i];
 printf(errors?"[NOT PASSED] checking keys: #err=%zu\n":"[PASSED] checking keys (%zu errors)\n",errors);
 if(!keys_only){sort_vals_of_same_key(goldkey,goldval,seg,num);sort_vals_of_same_key(goldkey,outval,seg,num);size_t ve=0;for(int i=0;i<elements;i++)ve+=goldval[i]!=outval[i];errors+=ve;printf(ve?"[NOT PASSED] checking vals: #err=%zu\n":"[PASSED] checking vals (%zu errors)\n",ve);}
 if(const char* path=getenv("SEGSORT_DUMP_DIR")){
  std::string f=std::string(path)+(keys_only?"/keys.bin":"/pairs.bin");
  FILE *fp=fopen(f.c_str(),"wb");if(!fp)throw std::runtime_error("dump open");
  fwrite(output.data(),4,elements,fp);if(!keys_only)fwrite(outval.data(),8,elements,fp);fclose(fp);
 }
 return errors==0;
}
int main(int argc,char**argv){
 int elements=1UL<<25;if(const char*s=getenv("SEGSORT_VALIDATION_ELEMENTS"))elements=atoi(s);
 if(elements<=0)return 2;
 iris::Platform platform;{GpuLock lock;platform.init(&argc,&argv,1);}
 int devices=0;iris_device_count(&devices);for(int d=0;d<devices;d++){int b=0;size_t z=0;iris_device_info(d,iris_backend,&b,&z);printf("IRIS backend=%d HIP=%d\n",b,iris_hip);}
 printf("The number of elements is %d\nRunning key only test\n",elements);
 bool ok=segsort(elements,true);printf("\nRunning key-value test\n");ok=segsort(elements,false)&&ok;
 {GpuLock lock;platform.finalize();}return ok?0:1;
}
