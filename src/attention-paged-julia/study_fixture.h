#pragma once
#include <vector>
#include <cmath>
#include <cassert>
struct AttentionFixture {
 int sequences,heads,head,maxlen,pages,style;int block=16;float scale,kvscale;
 std::vector<int>lengths,tables;
 AttentionFixture(int ci):sequences(ci==0?2:ci==1?3:1),heads(ci==1?4:2),head(ci==0?32:ci==1?64:128),maxlen(ci==0?33:ci==1?129:257),pages(ci==0?4:ci==1?16:24),style(ci){
  scale=1.f/std::sqrt(float(head));kvscale=std::pow(float(head),-.5f);
  lengths=ci==0?std::vector<int>{1,33}:ci==1?std::vector<int>{15,16,129}:std::vector<int>{257};
  int blocks=(maxlen+15)/16;tables.resize(sequences*blocks);for(int s=0;s<sequences;s++)for(int b=0;b<blocks;b++)tables[s*blocks+b]=pages-1-(3*b+5*s)%pages;
 }
 void override_inputs(std::vector<float>&q,std::vector<float>&k,std::vector<float>&v,std::vector<float>&alibi)const{
  if(style==1){
   for(int s=0;s<sequences;s++)for(int h=0;h<heads;h++)for(int d=0;d<head;d++)q[(s*heads+h)*head+d]=float(.7*std::sin((d+1)*.31+s*.2+h*.7));
   for(int p=0;p<pages;p++)for(int h=0;h<heads;h++)for(int d=0;d<head;d++)for(int t=0;t<16;t++){
    k[((p*heads+h)*(head/4)+d/4)*16*4+t*4+d%4]=float(.4*std::sin((p+1)*.2+(h+1)*.7+(d+1)*.11+(t+1)*.19));
    v[((p*heads+h)*head+d)*16+t]=float(.8*std::cos((p+1)*.13+(h+1)*.23+(d+1)*.29+(t+1)*.17));
   }
   float slopes[4]={0,.003f,-.004f,.01f};for(int h=0;h<heads;h++)alibi[h]=slopes[h];
  }
  if(style==2){
   for(int p=0;p<pages;p++)for(int h=0;h<heads;h++)for(int d=0;d<head;d++)for(int t=0;t<16;t++)v[((p*heads+h)*head+d)*16+t]=float(.02*(d%7)-.04+.01*h);
   std::fill(alibi.begin(),alibi.end(),0.f);
  }
 }
 std::vector<double>oracle(const std::vector<float>&q,const std::vector<float>&k,const std::vector<float>&v,const std::vector<float>&alibi)const{
  std::vector<double>out(q.size());int maxblocks=(maxlen+15)/16;
  for(int s=0;s<sequences;s++)for(int h=0;h<heads;h++){
   int L=lengths[s];std::vector<double>logits(L);double m=-INFINITY;
   for(int t=0;t<L;t++){int p=tables[s*maxblocks+t/16],off=t%16;double dot=0.;for(int d=0;d<head;d++)dot+=double(q[(s*heads+h)*head+d])*k[((p*heads+h)*(head/4)+d/4)*16*4+off*4+d%4];logits[t]=dot*double(scale)+double(alibi[h])*(t-L+1);m=std::max(m,logits[t]);}
   double norm=1e-6;for(double&x:logits){x=std::exp(x-m);norm+=x;}
   for(int d=0;d<head;d++){double sum=0.;for(int t=0;t<L;t++){int p=tables[s*maxblocks+t/16];sum+=logits[t]*double(v[((p*heads+h)*head+d)*16+t%16]);}out[(s*heads+h)*head+d]=sum/norm;}
  }
  return out;
 }
};
