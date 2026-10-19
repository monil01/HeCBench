#!/usr/bin/env python3
"""Independent Float64 conditional-Pearson and complete CCS selection/merge oracle."""
import math,struct,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
def f(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def pearson(x,y,mask):
 a=[u for u,t in zip(x,mask)if t];b=[v for v,t in zip(y,mask)if t]
 if not a:return math.nan
 mx=sum(a)/len(a);my=sum(b)/len(b);xx=sum((u-mx)**2 for u in a);yy=sum((v-my)**2 for v in b)
 return abs(sum((u-mx)*(v-my)for u,v in zip(a,b))/math.sqrt(xx*yy))if xx*yy>0 else math.nan
def run(c):
 n,D,m=c['n'],c['D'],c['maxbcn'];gene=[c['gene'][i*(D+1):i*(D+1)+D]for i in range(n)];means=[c['gene'][i*(D+1)+D]for i in range(n)];thr=c['thr'];margin=1.;corchecks=0
 def corr(mask,k,i):
  nonlocal margin,corchecks
  r=pearson(gene[k],gene[i],mask);nr=pearson(gene[k],gene[i],[not x for x in mask]);corchecks+=2
  for x in[r,nr]:
   if math.isfinite(x):margin=min(margin,abs(x-thr))
  return r,nr
 best=[dict(sample=[0]*D,data=[0]*n,score=1.,dc=0,sc=0)for _ in range(m)];tmp=[dict(sample=[0]*D,data=[0]*n)for _ in range(m)];ties=0
 for k in range(m):
  for i in range(k+1,n):
   masks=[[False]*D for _ in range(3)]
   for j in range(D):
    x=f(gene[k][j]-means[k]);y=f(gene[i][j]-means[i])
    if x>=0 and y>=0:v=0
    elif x<0 and y<0:v=1
    elif x*y<0:v=2
    else:raise AssertionError('Unassignedsourceclassification mean-tie outsidefixture')
    masks[v][j]=True
   for mask in masks:
    wid=sum(mask)
    if wid<=10:continue
    r,nr=corr(mask,k,i)
    if not r>thr:continue
    tot=1;dif=int(nr>thr);membership=[0]*n;membership[k]=membership[i]=1;count=2
    for l in range(n):
     if l in(k,i):continue
     ttot=tdif=0
     for li in range(n):
      if membership[li]:
       rr,nn=corr(mask,l,li)
       if rr>thr:ttot+=1
       else:ttot=0;break
       tdif+=int(nn>thr)
     if ttot>0:membership[l]=1;count+=1;tot+=ttot;dif+=tdif
    score=f(dif/tot)if tot else 1.
    tmp[k]=dict(sample=[49 if t else 48 for t in mask],data=[49 if t else 48 for t in membership])
    if score<.01 and count>10:
     if best[k]['dc']<count:best[k]=dict(sample=tmp[k]['sample'].copy(),data=tmp[k]['data'].copy(),score=score,dc=count,sc=wid)
     elif best[k]['dc']==count and best[k]['sample']!=tmp[k]['sample']:ties+=1
 def flatten(bs,ts=None):
  v=[]
  for key in['sample','data','score','dc','sc']:
   for b in bs:v.extend(b[key]if isinstance(b[key],list)else[b[key]])
  if ts is not None:
   for key in['sample','data']:
    for b in ts:v.extend(b[key])
  return v
 raw=flatten(best,tmp);mismatches=sum(b['dc']>10 and(b['sample']!=t['sample']or b['data']!=t['data'])for b,t in zip(best,tmp))
 for k in range(m):
  if best[k]['score']>=.01:continue
  for kk in range(k+1,m):
   if best[kk]['score']>=.01:continue
   data=[a==49 or b==49 for a,b in zip(best[k]['data'],best[kk]['data'])];common=sum(a==49 and b==49 for a,b in zip(best[k]['data'],best[kk]['data']));union=sum(data);observed=f(f(common/union)*100)if union else math.nan
   if observed>=c['overlap'] and observed<100:
    mask=[a==49 or b==49 for a,b in zip(best[k]['sample'],best[kk]['sample'])];dif=un=total=mintotal=0
    for i in range(n):
     if not data[i]:continue
     for l in range(i+1,n):
      if not data[l]:continue
      r,nr=corr(mask,i,l);a=r>thr;b=nr>thr;total+=a;mintotal+=b;dif+=a and b;un+=a or b
    score=f(dif/un)if un>0 and total>mintotal else 1.
   elif observed==100:score=best[k]['score']
   else:score=1.
   if score<.01:
    for key,count in [('sample','sc'),('data','dc')]:
     for j in range(len(best[k][key])):
      if best[kk][key][j]==49 and best[k][key][j]==48:best[k][key][j]=49;best[k][count]+=1
    best[kk]['score']=1.;best[k]['score']=score
 text=''
 for b in best:
  if b['score']<.01 and b['dc']>10 and b['sc']>10:
   text+=''.join(str(i)+' 'for i,x in enumerate(b['data'])if x==49)+'\n'+''.join(str(i)+' 'for i,x in enumerate(b['sample'])if x==49)+'\n\n'
 c['expected']=raw+flatten(best)+[len(text)]+list(text.encode())+[-777.]*(16*4)+[165.]*(16*4)
 c['oracle_checks']=corchecks;c['minimum_threshold_margin']=margin;c['selected_vs_tmp_mismatching_seeds']=mismatches;c['equal_count_distinct_mask_candidates']=ties;c['printed_biclusters']=text.count('\n\n')
 return c
def main():
 cs=[]
 for k,(n,D,m,overlap)in enumerate([(12,32,4,100.),(17,64,5,50.),(33,200,7,100.)]):
  gene=[];half=D//2
  for i in range(n):
   row=[]
   for j in range(D):
    if i<12:
     x=(8+((j*7)%13))*(1+i/16)+i/8 if j<half else -16+.5*(-1 if ((j-half)&(i+1)).bit_count()%2 else 1)
    elif i in[12,13]and D>=64:
     lo=0 if i==12 else half//2;hi=half//2 if i==12 else half
     x=(8+((j*7)%13))*(1+i/16)+i/8 if lo<=j<hi else -16+.5*(-1 if (j&(i+1)).bit_count()%2 else 1)
    else:x=(8 if j<half else -16)+(-1 if (j&(i+1)).bit_count()%2 else 1)*2
    row.append(f(x))
   acc=0.
   for x in row:acc=f(acc+x)
   mean=f(acc/D);assert all(x!=mean for x in row);gene.extend(row+[mean])
  c=run(dict(n=n,D=D,maxbcn=m,repeat=3,thr=f(.9),overlap=f(overlap),gene=gene));assert c['minimum_threshold_margin']>1e-4;assert c['printed_biclusters']>0;cs.append(c)
 raw=bytearray(struct.pack('<I',len(cs)))
 for c in cs:
  raw+=struct.pack('<6I2f',c['n'],c['D'],c['maxbcn'],c['repeat'],len(c['gene']),len(c['expected']),c['thr'],c['overlap']);raw+=struct.pack('<%df'%len(c['gene']),*c['gene']);raw+=struct.pack('<%dd'%len(c['expected']),*c['expected'])
 (P/'fixtures.bin').write_bytes(raw);report=[{k:v for k,v in c.items()if k not in['gene','expected']}for c in cs];(P/'fixture_manifest.json').write_text(json.dumps({'generator':'IndependentFloat64conditionalPearson/CCSaugmentation/strictlargestcandidate/sourceorderedmerge/printedindices','cases':report,'sha256':hashlib.sha256(raw).hexdigest()},indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
