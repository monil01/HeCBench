#!/usr/bin/env python3
"""CPU-only full saved-array/source audit, not an additional GPU measurement."""
import argparse,pathlib,json,hashlib,mmap,struct,math,sys,datetime
import numpy as np
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]
def sha(p):
 with pathlib.Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bytes_equal(a,b,limit=None):
 with pathlib.Path(a).open('rb')as x,pathlib.Path(b).open('rb')as y:
  remaining=limit
  while remaining is None or remaining:
   n=4*1024*1024 if remaining is None else min(remaining,4*1024*1024);u=x.read(n);v=y.read(n)
   assert u==v,'actual saved bytes differ'
   if not u:assert remaining is None;return True
   if remaining is not None:remaining-=len(u)
 return True
class Reader:
 def __init__(self,b,offset):self.b=b;self.pos=offset
 def take(self,fmt):v=struct.unpack_from('<'+fmt,self.b,self.pos)[0];self.pos+=struct.calcsize('<'+fmt);return v
 def array(self,dtype,n):a=np.frombuffer(self.b,dtype=dtype,count=n,offset=self.pos);self.pos+=a.nbytes;return a
MCS_CACHE={}
def chordal(rows):
 key=tuple(int(x)for x in rows)
 if key in MCS_CACHE:return MCS_CACHE[key]
 n=len(key);weights=[0]*n;rank=[0]*n;seen=0
 for k in range(n):
  v=max((i for i in range(n)if not(seen>>i&1)),key=lambda i:weights[i]);old=key[v]&seen
  if old:
   parent=max((i for i in range(n)if old>>i&1),key=lambda i:rank[i]);assert not((old&~(1<<parent))&~key[parent]),'nonchordal saved host graph'
  rank[v]=k;seen|=1<<v
  for i in range(n):weights[i]+=int(bool(key[v]>>i&1))
 MCS_CACHE[key]=True;return True
STATS={'states':0,'graphs':0,'finite_float_fields':0,'empty_component_states':0}
def read_state(r):
 n=r.take('i');p=r.take('i');L=r.take('i');assert n==150 and p==50 and 0<L<=n
 plp=r.take('f');alpha=r.take('f');assert math.isfinite(plp)and math.isfinite(alpha);STATS['finite_float_fields']+=2
 xi=r.array('<i2',n);assert np.all((xi>=0)&(xi<L));counts=np.bincount(xi,minlength=L);STATS['empty_component_states']+=int(np.any(counts==0));graphs=[];pll=[]
 for l in range(L):
  score=r.take('f');assert math.isfinite(score);pll.append(score);STATS['finite_float_fields']+=1;v=r.take('i');assert v==p;rows=r.array('<u8',p);assert np.all((rows>>np.uint64(p))==0)
  for i,row in enumerate(rows):
   assert not(int(row)>>i&1)
   for j in range(p):assert(int(row)>>j&1)==(int(rows[j])>>i&1)
  chordal(rows);graphs.append(tuple(int(x)for x in rows));STATS['graphs']+=1
 STATS['states']+=1;return {'L':L,'xi':xi,'plp':plp,'pll':pll,'alpha':alpha,'graphs':graphs}
def read_list(r):
 n=r.take('i');p=r.take('i');M=r.take('i');assert n==150 and p==50 and M in(2,20);ee=p*(p-1)//2
 LL=r.array('<i2',M);xi=r.array('<i2',M*n).reshape(M,n);edges=r.array('<i2',M*n*ee).reshape(M,n,ee);scores=r.array('<f4',M);index=r.take('i');minimum=r.take('f');assert np.isfinite(scores).all()and math.isfinite(minimum)and 0<=index<M;models=[]
 for m in range(M):
  L=int(LL[m])
  if L==-1:
   assert np.all(xi[m]==-1)and np.all(edges[m]==-1);continue
  assert 0<L<=n and np.all((xi[m]>=0)&(xi[m]<L));assert np.all((edges[m]==0)|(edges[m]==1));modelgraphs=[];model_edges=[]
  for l in range(L):
   indices=np.flatnonzero(xi[m]==l);assert len(indices)>0,'source writer assumes occupied component';e=edges[m,int(indices[0])];assert np.all(edges[m,indices]==e),'saved model has inconsistent graph across component members';rows=[0]*p;k=0
   for i in range(p-1):
    for j in range(i+1,p):
     if e[k]:rows[i]|=1<<j;rows[j]|=1<<i
     k+=1
   chordal(rows);modelgraphs.append(tuple(rows));model_edges.append(e)
  models.append({'L':L,'xi':xi[m],'edges':model_edges,'graphs':modelgraphs,'score':float(scores[m])})
 return {'capacity':M,'active_models':models,'minimum':minimum,'index':index}
def writer_check(path,modes):
 tokens=path.read_text().split();q=0
 def get():
  nonlocal q
  v=tokens[q];q+=1;return v
 assert int(get())==150 and int(get())==50
 for model in modes['active_models']:
  L=int(get());assert L==model['L'];score=float(get());assert math.isfinite(score);true=model['score'];bound=.5*10**(math.floor(math.log10(abs(true)))-5)if true else 0;assert abs(score-true)<=bound+1e-7,'source six-significant-digit score serialization'
  assert np.array_equal(np.array([int(get())for _ in range(150)]),model['xi'])
  for l in range(L):assert np.array_equal(np.array([int(get())for _ in range(1225)]),model['edges'][l])
  for l in range(L):
   rows=model['graphs'][l];nc=int(get());assert 0<nc<=50;cliques=[]
   for c in range(nc):
    dim=int(get());vertices=[int(get())for _ in range(dim)];assert 0<dim<=50 and len(set(vertices))==dim and all(0<=x<50 for x in vertices)
    mask=sum(1<<v for v in vertices)
    for v in vertices:assert(mask&~(1<<v))&~rows[v]==0,'writer clique not complete'
    assert not any(all(rows[v]>>u&1 for v in vertices)for u in range(50)if not(mask>>u&1)),'writer clique not maximal';cliques.append(set(vertices))
   ne=int(get());tree=[(int(get()),int(get()))for _ in range(ne)];assert all(0<=a<nc and 0<=b<nc and a!=b for a,b in tree)
   ns=int(get());separators=[]
   for c in range(ns):
    dim=int(get());v=[int(get())for _ in range(dim)];assert 0<=dim<=50 and len(set(v))==dim and all(0<=x<50 for x in v);separators.append(set(v))
   assert ns==ne and all(s==cliques[a]&cliques[b]for s,(a,b)in zip(separators,tree)),'writer separator/clique-tree inconsistency'
 assert q==len(tokens),'unconsumed source writer fields';return {'tokens_checked':q,'models':len(modes['active_models']),'complete_text_semantics_match_saved_modes':True}
def typed_capture(path,layout,writer):
 with path.open('rb')as f:
  b=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);rd=Reader(b,0);assert rd.take('i')==150 and rd.take('i')==50;X=rd.array('<f4',7500);assert np.isfinite(X).all();tokens=(P/'original_native/DATA/f9_n150_p50.txt').read_text().split();assert list(map(int,tokens[:2]))==[150,50];gold=np.array(tokens[2:],dtype=np.float32);assert X.tobytes()==gold.tobytes(),'actualsourceX differsfromindependenttextparse'
  prefix=2*(layout['host_input_shorts']+layout['device_input_shorts']);offset=prefix+2*layout['device_output_shorts'];rd.pos=offset;iterations=0
  while struct.unpack_from('<q',b,rd.pos)[0]==iterations+1:
   k=rd.take('q');scores=[rd.take('f')for _ in range(3)];assert all(math.isfinite(x)for x in scores);modes=rd.take('i');cases=rd.take('i');total=rd.take('q');assert modes>0 and cases>=0 and total>=0;read_state(rd);iterations+=1
  final_k=rd.take('q');final_total=rd.take('q');assert final_k==0 and final_total==12504300;final_states=[read_state(rd)for _ in range(3)];modes=read_list(rd);features=read_list(rd);guard=rd.array('<i2',256);assert np.all(guard==0x5a5a)and rd.pos==len(b);text=writer_check(writer,modes)
  result={'iterations':iterations,'final_source_k':final_k,'terminal_restart_counter_reset':True,'full_search_model_proposals':final_total,'typed_state_checks':dict(STATS),'unique_chordal_graphs_checked':len(MCS_CACHE),'final_state_components':[s['L']for s in final_states],'active_saved_modes':len(modes['active_models']),'active_saved_feature_models':len(features['active_models']),'source_writer':text,'all_typed_bytes_consumed':True,'physical_redzone_words':256,'input_X_independent_text_parse_bitexact':True}
  # mmap views remain live until function returns; OS closes the map after refcount release.
  return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--result',type=pathlib.Path,required=True);ap.add_argument('--capture-root',type=pathlib.Path,default=pathlib.Path('/tmp/hecbench-portable-infrastructure-sss-captures'));a=ap.parse_args();r=json.loads(a.result.read_text());assert r['matched_performance_valid']and r['reference_agreement']and all(r['variant_verified'].values());report={'benchmark':'sss','result_path':str(a.result),'result_sha256':sha(a.result),'source_bindings':[],'artifacts':[],'dependency_failures':[],'primary_acceptance_changed':False,'GPU_executions':0,'timings_changed':False,'additional_typed_checks_predeclared_in_detail':False,'additional_typed_check_scope':'Postexecution corroboration of fullsavedhoststate/textoutputs, not a relaxedacceptance or an independentGWishartposterior implementation. PrimaryruntimeMCS+exactallbytes contract unchanged.'};first=None;first_sha=None;first_meta=None;first_writer=None
 for model,m in r['models'].items():
  assert m['complete']and m['provenance_valid']and len(m['trials'])==5
  for path,expected in m['adapter_file_hashes'].items():
   p=pathlib.Path(path);actual=sha(p)if p.is_file()else None;item={'model':model,'path':path,'expected_sha256':expected,'actual_sha256':actual,'pass':actual==expected};report['source_bindings'].append(item)
   if actual!=expected:report['dependency_failures'].append(item)
  for row in m['trials']:
   art=row['output_artifact'];saved=pathlib.Path(art['saved_path']);actual=sha(saved);assert actual==art['sha256'];assert saved.stat().st_size==242952380 and art['count']==242952380 and art['dtype']=='uint8';source=pathlib.Path(art['path']);assert bytes_equal(saved,source);base=source.parent;layout=json.loads((base/'layout.json').read_text());assert layout['checked_gates']==12469275 and layout['guards']==256 and layout['device_output_shorts']==24938550;meta=base/'records_i64.bin';writer=base/'f9_n150_p50_modes_GPU.txt'
   if first is None:first=saved;first_sha=actual;first_meta=meta;first_writer=writer;report['typed_first_artifact_checks']=typed_capture(saved,layout,writer)
   else:assert actual==first_sha and bytes_equal(saved,first);assert bytes_equal(meta,first_meta)and bytes_equal(writer,first_writer)
   assert bytes_equal(saved,first,181368048)
   report['artifacts'].append({'model':model,'trial':row['trial'],'path':str(saved),'source_original_capture':str(source),'bytes':saved.stat().st_size,'sha256':actual,'full_byte_comparison_to_reference_pass':True,'actual_source_capture_copy_exact':True,'input_prefix_bytes_checked':181368048,'records_sha256':sha(meta),'writer_sha256':sha(writer),'all_runtime_MCS_gates':12469275,'all_inactive_output_tail_words':12469275,'guard_words':256,'typed_mathematical_checks':'firstfulltypeddecode reused onlyaftercompleteartifactbyteequality/fullhash; no summary-substitution'})
 assert not report['dependency_failures'];assert len(report['artifacts'])==20
 report.update(status='pass',artifact_count=20,total_logical_primary_bytes=sum(x['bytes']for x in report['artifacts']),all_20_complete_values_and_inputs_compared=True,all20_source_text_and_graph_record_sidecars_exact=True,independent_GPU_predicate_checks_retained_trials=20*12469275,native_source_reused_scientific_CPU_controller=True,independent_full_GWishart_posterior_quality_certificate=False,numpy_version=np.__version__,python_version=sys.version,script_sha256=sha(pathlib.Path(__file__)),timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat());dest=P/'final_saved_full_audit.json';dest.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':report['status'],'artifacts':20,'bytes':report['total_logical_primary_bytes'],'typed_checks':report['typed_first_artifact_checks'],'bindings':len(report['source_bindings']),'output':str(dest)}))
if __name__=='__main__':main()
