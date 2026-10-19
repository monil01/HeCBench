"""Full saved-array replay independent of the native and Julia study adapters."""
from pathlib import Path
from array import array
import argparse,hashlib,json,math,sys
p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();P=Path(__file__).resolve().parent;assert sys.byteorder=='little'
def read(q):
 z=array('d')
 with q.open('rb')as f:z.fromfile(f,q.stat().st_size//8)
 return z
fixture=read(P/'fixture.bin');gold=read(P/'oracle.bin');assert fixture[0]==2;cases=[];k=1;prefix=array('d',[2]);guards=lambda n,v:array('d',[v])*n
for ci in range(2):
 head=fixture[k:k+11];nc,ns,ng,nt,mt=map(int,head[:5]);k+=11;counts=list(map(int,fixture[k:k+ng]));k+=ng;geom=[]
 for z in range(5):geom.append(fixture[k:k+nt]);k+=nt
 samples=fixture[k:k+nt*ns];k+=nt*ns;chunks=geom+[array('d',[0])*n for n in [nc,nt,ns*nc,ns*nc,ng*ns,ng*ns,ng*ns,mt*ns]];pre=array('d');pre.extend(head);pre.extend(counts)
 for ix,v in enumerate(chunks):
  guard=-9001 if ix==9 else -9001.25;pre.extend([guard]*32);pre.extend(v);pre.extend([guard]*32)
 prefix.extend(pre);cases.append((head,counts,geom,samples,chunks))
assert k==len(fixture)
for head,counts,geom,samples,chunks in cases:
 ns=int(head[1]);current=array('d',chunks[-1]);begin=0
 for stop in counts:
  size=(stop-begin)*ns;current[:size]=samples[begin*ns:stop*ns];prefix.extend([-9001.25]*32);prefix.extend(current);prefix.extend([-9001.25]*32);begin=stop
assert len(prefix)==128743
j=json.loads(a.result.read_text());rows=[];cache={}
for model,spec in j['models'].items():
 for trial in spec.get('trials',[]):
  art=trial['output_artifact'];path=Path(art['saved_path']);digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest();assert digest==art['sha256'];assert path.stat().st_size==art['count']*8
  if digest not in cache:
   x=read(path);assert x[:len(prefix)]==prefix;k=len(prefix);g=0;maxerr=0.;maxbudget=0.;floatchecks=0;indexchecks=0
   def out(size,guard=-9001.25):
    global k
    assert all(z==guard for z in x[k:k+32]);v=x[k+32:k+32+size];assert all(z==guard for z in x[k+32+size:k+64+size]);k+=size+64;return v
   def compare(got,ref,exact=False):
    global maxerr,maxbudget,floatchecks,indexchecks
    assert len(got)==len(ref)
    for v,e in zip(got,ref):
     assert math.isfinite(v)
     if exact:assert v==e;indexchecks+=1
     else:
      error=abs(v-e);budget=5e-5+2e-4*abs(e);assert error<=budget;maxerr=max(maxerr,error);maxbudget=max(maxbudget,error/budget);floatchecks+=1
   for head,counts,geom,samples,chunks in cases:
    nc,ns,ng,nt,mt=map(int,head[:5]);curv=out(nc);compare(curv,gold[g:g+nc]);g+=nc;half=out(nt);compare(half,gold[g:g+nt]);g+=nt
    expectedctr=array('d',[0])*(ng*ns);expectedstr=expectedctr[:];expectedstk=expectedctr[:];begin=0
    for group,stop in enumerate(counts):
     num=out(ns*nc);compare(num,gold[g:g+ns*nc]);g+=ns*nc;stt=out(ns*nc);compare(stt,gold[g:g+ns*nc]);g+=ns*nc
     expectedctr[group*ns:(group+1)*ns]=gold[g:g+ns];g+=ns;expectedstr[group*ns:(group+1)*ns]=gold[g:g+ns];g+=ns;expectedstk[group*ns:(group+1)*ns]=gold[g:g+ns];g+=ns
     ctr=out(ng*ns,-9001);strength=out(ng*ns);stack=out(ng*ns);compare(ctr,expectedctr,True);compare(strength,expectedstr);compare(stack,expectedstk)
     for t in range(ns):
      vals=num[t*nc:(t+1)*nc];assert all(v==-1 or 0<=v<=1.0001 for v in vals);best=max([0.]+list(vals));ix=vals.index(best)if best>0 else -1;assert ctr[group*ns+t]==ix;assert strength[group*ns+t]==best;assert stack[group*ns+t]==(stt[t*nc+ix]if ix>=0 else 0.)
     begin=stop
    for v in geom:compare(out(nt),v,True)
    compare(out(nc),curv,True);compare(out(nt),half,True);current=array('d',chunks[-1]);begin=0
    for stop in counts:current[:(stop-begin)*ns]=samples[begin*ns:stop*ns];begin=stop
    compare(out(mt*ns),current,True)
   assert g==len(gold)and k==len(x);cache[digest]={'independent_oracle_values_consumed':g,'floating_comparisons_including_full_cumulative_outputs':floatchecks,'exact_internal_indices_readonly_comparisons':indexchecks,'max_abs_error':maxerr,'max_fraction_of_predeclared_budget':maxbudget}
  rows.append({'model':model,'trial':trial['trial'],'verdict':'pass','full_artifact_sha256_rechecked':digest,'math_reused_only_for_identical_complete_hash':digest in [z['full_artifact_sha256_rechecked']for z in rows]})
a.output.write_text(json.dumps({'run_id':j['run_id'],'exact_actual_external_input_prefix_words':len(prefix),'rows':rows,'distinct_complete_numerical_replays':cache,'scope':'Full source c/h producer outputs and num/stt internalconsumerarrays captured readonly underorderedsourcepipeline; not falselydefinedexternalbitidentity. ActualsampleH2Dinputs complete/exact. IndependentF64 allsurfaces, strictfirstpositiveargmaxselfconsistency/indices,guards/read-only.'},indent=2)+'\n')
