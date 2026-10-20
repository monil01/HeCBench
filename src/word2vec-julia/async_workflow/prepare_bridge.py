#!/usr/bin/env python3
from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'observed_gpu.inc').read_text();s=s[:s.index('__global__')]+s[s.index('#define cudaCheck'):]
s=s.replace('real * d_syn0 = NULL;','real * d_syn0 = NULL;\nreal* target_exp=nullptr;')
s=s.replace('void cleanUpGPU(){\n  observe_final();','void cleanUpGPU(){\n  observe_final();\n  cudaCheck(cudaFree(target_exp));')
s=s.replace('  free(h_expTable);','  cudaCheck(cudaMalloc((void**)&target_exp,4000));\n  cudaCheck(cudaMemcpy(target_exp,h_expTable,4000,cudaMemcpyHostToDevice));\n  free(h_expTable);')
s=s.replace('    device_memset<<<syn1neg_size / maxThreadsPerBlock + 1, maxThreadsPerBlock>>>(d_syn1neg, syn1neg_size);','    broker(0,0,0,syn1neg_size / maxThreadsPerBlock + 1);')
a='''  device_cbow<<<numBlock,BLOCK_SIZE, shared_mem_usage >>>(
    sentence_num, layer1_size, layer1_size_aligned, window,
    negative, table_size, vocab_size, d_sen, d_table, d_syn0, d_syn1neg, d_random, trace_data, slot);'''
assert a in s;s=s.replace(a,'  broker(1,sentence_num,slot,numBlock);');(P/'target_gpu.inc').write_text(s)
s=(P/'study_native.cu').read_text();s=s.replace('#include "observed_gpu.inc"','#include "target_gpu.inc"');s=s.replace('void observe_initial();void observe_final();','void observe_initial();void observe_final();\nvoid broker(int kind,int ns,int slot,int blocks);')
a='device_cbow<<<1,128,shared_mem_usage>>>(1,layer1_size,layer1_size_aligned,window,negative,table_size,vocab_size,d_sen,d_table,d_syn0,d_syn1neg,d_random,trace_data,0);';assert a in s;s=s.replace(a,'broker(1,1,0,1);')
s=s.replace(' snapshot("final_syn0.f32"',' snapshot("final_syn0.f32"') # keep exact observer unchanged
s=s.replace('void observe_final(){ck(cudaDeviceSynchronize());Pause pause;','void observe_final(){ck(cudaDeviceSynchronize());Pause pause;snapshot("target_exptable.f32",target_exp,4000);')
s=s.replace('int main(int argc,char**argv)','int source_workflow_run(int argc,char**argv)')
s+='''
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
struct Request {int kind,slot,ns,blocks,d,a,w,neg,t,v;void*s0;void*s1;void*sen;void*rng;void*table;void*trace;void*counter;void*ex;};
struct Pending{Request r;bool ack=false;std::condition_variable cv;};
std::mutex qmutex;std::deque<Pending*> queue;std::thread workflow;std::atomic<bool> finished{false};std::atomic<int> workflow_rc{-999};
void broker(int kind,int ns,int slot,int blocks){unsigned int*counter;ck(cudaMemcpyFromSymbol(&counter,trace_execution_counter,sizeof(counter)));Pending pending;pending.r={kind,slot,ns,blocks,layer1_size,layer1_size_aligned,window,negative,table_size,vocab_size,d_syn0,d_syn1neg,d_sen,d_random,d_table,trace_data,counter,target_exp};std::unique_lock<std::mutex>lock(qmutex);queue.push_back(&pending);pending.cv.wait(lock,[&]{return pending.ack;});}
extern "C" int wf_start(const char*base){if(workflow.joinable())return -1;finished=false;std::string path=base;workflow=std::thread([path]{char arg0[]="workflow";std::vector<char>p(path.begin(),path.end());p.push_back(0);char*av[]={arg0,p.data()};workflow_rc=source_workflow_run(2,av);finished=true;});return 0;}
extern "C" int wf_poll(Request*out){std::lock_guard<std::mutex>lock(qmutex);if(!queue.empty()){*out=queue.front()->r;return 1;}return finished?-1:0;}
extern "C" void wf_ack(){std::lock_guard<std::mutex>lock(qmutex);auto*p=queue.front();queue.pop_front();p->ack=true;p->cv.notify_one();}
extern "C" int wf_join(){if(workflow.joinable())workflow.join();return workflow_rc;}
'''
(P/'study_bridge.cu').write_text(s)
# Separate checker version; historical accepted native source remains frozen.
s=(P/'check_properties.py').read_text().replace("guards=array('guards.bin','u1',5*512)","guards=array('guards.bin','u1',6*512)").replace("all_5_actual_allocation_guards","all_6_actual_allocations_guards_5source_plus_exp_mapping")
s=s.replace(" require('full_model_finite'", " require('full_model_finite'")
a=" guards=array('guards.bin'";idx=s.index(a);s=s[:idx]+" target_exp=array('target_exptable.f32','<f4',1000);require('mapped_exptable_exact_readonly',np.array_equal(target_exp,actual_exp))\n"+s[idx:];(P/'check_target_properties.py').write_text(s)
