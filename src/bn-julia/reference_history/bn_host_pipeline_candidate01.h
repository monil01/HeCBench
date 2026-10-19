#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <algorithm>
#include "projected_data.h"
#include "reference.h"
#include "source_host_functions.inc"
int study_c=0;
extern "C" {
int bn_nodes(){return NODE_N;}
int bn_data_count(){return NODE_N*DATA_N;}
int bn_scores_count(){return NODE_N*sizepernode;}
int bn_size(){return sizepernode;}
int*bn_data(){return data;}
float*bn_lg(){return LG;}
unsigned char*bn_mask(int node){return (unsigned char*)orders[node];}
float bn_score(){return score;}
int bn_choose(int n,int k){return C(n,k);}
void bn_reset(int which){
 static int original_data[DATA_N*NODE_N];static bool first=true;if(first){std::copy(data,data+DATA_N*NODE_N,original_data);first=false;}
 if(which==0)std::copy(original_data,original_data+DATA_N*NODE_N,data);
 if(which==1)for(int t=0;t<DATA_N;t++)for(int n=0;n<NODE_N;n++)data[t*NODE_N+n]=((t>>(n%4))^(t/(11+n))^(n%3==0?t/3:0))&1;
 if(which==2)std::fill(data,data+DATA_N*NODE_N,0);
 preScore=-99999999999.f;score=0.f;study_c=0;std::fill(maxScore,maxScore+HIGHEST,0.f);maxScore[0]=-999999999.f;memset(preGraph,0,sizeof(preGraph));memset(bestGraph,0,sizeof(bestGraph));srand(2);initial();Pre_logGamma();
}
void bn_release(){free(localscore);free(localscore_ref);free(LG);localscore=localscore_ref=LG=nullptr;}
void bn_prepare_iteration(){for(int a=0;a<NODE_N;a++)for(int j=0;j<NODE_N;j++)orders[a][j]=preOrders[a][j];int tmp=rand()%6;for(int j=0;j<tmp;j++)genOrders();memset(graph,0,sizeof(graph));score=0.f;}
int bn_total(int node){int n=0;for(int i=0;i<NODE_N;i++)n+=orders[node][i];return C(n,4)+C(n,3)+C(n,2)+n+1;}
void bn_accept_node(int node,int blocks,const float*scores,const int*parents){
 float bestls=-99999999.f;int bestparent[5]={0};int bestpN=0;
 for(int i=0;i<blocks;i++)if(scores[i]>bestls){bestls=scores[i];int n=0;for(int j=0;j<4;j++){if(parents[i*4+j]<0)break;bestparent[j]=parents[i*4+j];n++;}bestpN=n;}
 if(bestls>-99999999.f){for(int i=0;i<bestpN;i++){if(bestparent[i]<=node)graph[node][bestparent[i]-1]=1;else graph[node][bestparent[i]]=1;}score+=bestls;}
}
void bn_finish_iteration(){
 ConCore();int &c=study_c;int tmp=0,j=0,a=0,b=0;float tmpd=0;
 #include "source_top3_selection.inc"
}
void bn_snapshot(double*out){int k=0;out[k++]=score;out[k++]=preScore;out[k++]=study_c;for(auto&row:orders)for(bool x:row)out[k++]=x;for(auto&row:graph)for(bool x:row)out[k++]=x;for(auto&row:preOrders)for(bool x:row)out[k++]=x;for(auto&row:preGraph)for(bool x:row)out[k++]=x;for(float x:maxScore)out[k++]=x;for(auto&g:bestGraph)for(auto&row:g)for(bool x:row)out[k++]=x;assert(k==573);}
}
