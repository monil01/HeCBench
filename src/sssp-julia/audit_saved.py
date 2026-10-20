#!/usr/bin/env python3
"""Independent full Dijkstra/initial GPU-prefix/private-state evidence audit; no GPU."""
import argparse,json,pathlib,hashlib
import numpy as np
P=pathlib.Path(__file__).resolve().parent
G=-19371337;INF=-2147483647;WHITE=16677217;GRAY0=16677219;GRAY1=16677220;BLACK=16677221
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def expected_prefix(case,limit):
 raw=np.fromfile(P/(case+'_graph_i32.bin'),dtype='<i4');n,e,source=map(int,raw[:3]);nodes=raw[3:3+2*n].reshape(n,2);edges=raw[3+2*n:].reshape(e,2);cost=np.full(n,INF,dtype='<i4');cost[source]=0;color=np.full(n,WHITE,dtype='<i4');q1=np.zeros(n,dtype='<i4');q1[0]=source;q2=np.zeros(n,dtype='<i4');s=np.array([1,0,0,0,GRAY0,0,0,0],dtype='<i4');color[source]=BLACK
 for j in range(int(nodes[source,0]),int(nodes[source].sum())):
  v,w=map(int,edges[j]);cost[v]=w;color[v]=GRAY0;q2[s[2]]=v;s[2]+=1
 s[0]=s[2];s[2]=0;s[6]+=1;s[4]=GRAY1;s[5]+=1
 while 0<s[0]<limit:
  qi,qo=(q1,q2)if s[5]%2==0 else(q2,q1)
  for k in range(int(s[0])):
   u=int(qi[k]);color[u]=BLACK;cur=int(cost[u])
   for j in range(int(nodes[u,0]),int(nodes[u].sum())):
    v,w=map(int,edges[j]);cand=cur+w;old=int(cost[v]);cost[v]=max(old,cand)
    if old<cand:
     oldcol=int(color[v]);color[v]=max(oldcol,int(s[4]))
     if oldcol!=s[4]:qo[s[2]]=v;s[2]+=1;assert s[2]<=n
  s[5]+=1;s[0]=s[2];s[2]=0;s[1]=0;s[4]=GRAY0 if s[5]%2==0 else GRAY1
 assert s[0]>0
 def guard(a):return np.r_[a.ravel(),np.full(17,G,dtype='<i4')].astype('<i4')
 return np.concatenate([guard(nodes),guard(edges),guard(cost),guard(color),guard(q1),guard(q2),guard(s)]),n,e

def audit(result):
 r=json.load(open(result));cfg=json.load(open(pathlib.Path(result).parent/'config.json'));contract=cfg['input_contract'];case=contract['asset_case'];prefix,n,e=expected_prefix(case,contract['switching_limit']);assert len(prefix)==r['input_prefix_count']==6*n+2*e+127;oracle=np.fromfile(P/(case+'_oracle_i32.bin'),dtype='<i4');rows=[];total=0;auxbytes=0
 for model,m in r['models'].items():
  for t in m.get('trials',[]):
   f=t['output_artifact']['saved_path'];x=np.fromfile(f,dtype='<i4');assert len(x)==len(prefix)+n+17 and np.array_equal(x[:len(prefix)],prefix);out=x[len(prefix):];assert np.array_equal(out[:n],oracle)and np.all(out[n:]==G)
   a=t['raw_private_artifact'];af=pathlib.Path(a['path']);ax=np.fromfile(af,dtype='<i4');assert len(ax)==a['count']==7*n+169
   cur=0;gpu=[]
   for size in [n,n,n,n,8]:
    v=ax[cur:cur+size];assert np.all(ax[cur+size:cur+size+17]==G);gpu.append(v);cur+=size+17
   host=[]
   for size in [n,n,n,8]:
    v=ax[cur:cur+size];assert np.all(ax[cur+size:cur+size+17]==G);host.append(v);cur+=size+17
   assert cur==len(ax);assert np.all(gpu[0]<=oracle);assert np.all(np.isin(gpu[1],[WHITE,GRAY0,GRAY1,BLACK]));assert np.all(host[0]==np.where(oracle==INF,WHITE,BLACK));assert all(np.all((q>=0)&(q<n))for q in [gpu[2],gpu[3],host[1],host[2]]);hs=host[3];gs=gpu[4];assert tuple(hs[:4])==(0,0,0,0) and hs[4]in (GRAY0,GRAY1)and hs[5]>0 and hs[6]==1 and hs[7]==0;assert gs[3]==0 and gs[4]in (GRAY0,GRAY1)and 0<gs[0]<=n and 0<=gs[2]<=n
   if contract['require_actual_final_GPU_complete_oracle']:assert np.array_equal(gpu[0],oracle) and t['last_stage_GPU']and t['final_GPU_cost_exact_independent_Dijkstra']
   rows.append(dict(model=model,trial=t['trial'],full_observable_exact=True,first_GPU_input_prefix_exact=True,prefix_CPU_independent_source_preparation_replay=True,source_preparation_replay_is_not_Dijkstra_oracle=True,all_guards_exact=True,actual_last_GPU_partial_cost_valid_walk_bound=True,actual_final_GPU_cost_exact=bool(contract['require_actual_final_GPU_complete_oracle']),artifact=dict(path=f,sha256=sha(f),bytes=pathlib.Path(f).stat().st_size),raw_private_artifact=dict(path=str(af),sha256=sha(af),bytes=af.stat().st_size),private_queue_order_identity_claim=False));total+=pathlib.Path(f).stat().st_size;auxbytes+=af.stat().st_size
 return dict(benchmark='sssp',status='pass',result=str(pathlib.Path(result).resolve()),result_sha256=sha(result),configuration_id=r['configuration_id'],oracle='CompleteCPUheapDijkstra+sourceexpectedNYRfile/Bellmanandpredecessorcertificate',case=case,rows=rows,primary_artifact_bytes=total,raw_private_artifact_bytes=auxbytes,numpy_version=np.__version__,no_GPU_execution=True,known_scope='FullsourceobservableInt32distanceexact; privateGPUintermediatecostandqueueorderretainedraw/propertieschecked,notcanonicalizedorclaimedidentical.')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('result');a.add_argument('--output',required=True);z=a.parse_args();v=audit(z.result);pathlib.Path(z.output).write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({k:v[k]for k in ['status','primary_artifact_bytes','raw_private_artifact_bytes']}))
