"""Full independent BFS from every actual device CSR readback; no expected-label asset use."""
from pathlib import Path
import json,struct,hashlib,collections,sys
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent;F=Path(sys.argv[1])if len(sys.argv)>1 else R/'.porting-state/portable_julia/runs/20261019T164405388398Z_cc/result.json';r=json.loads(F.read_text());plan=json.loads((P/'study_plan.json').read_text());records=[];bindings={}
for model,m in r['models'].items():
 assert m['provenance_valid']and m['status']=='native_check_pass'and len(m['trials'])==5
 for name,sha in m['adapter_file_hashes'].items():
  p=Path(name);assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;bindings[str(p.resolve().relative_to(R))]=sha
 for t in m['trials']:
  art=t['output_artifact'];p=Path(art['saved_path']);raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha==art['sha256'];assert len(raw)==plan['artifact_count']*4;x=struct.unpack('<'+'i'*plan['artifact_count'],raw);offset=0;oracles=[];checks=[]
  for c in plan['cases']:
   n,e=c['n'],c['e'];header=x[offset:offset+7];offset+=7;assert header==(n,e,100,640,n,e,640)
   arrays=[]
   for k in ['idx_count','adj_count','idx_count','adj_count']:
    count=c[k];a=x[offset:offset+count];offset+=count;assert all(v==plan['guard']for v in a[:16]+a[-16:]);arrays.append(a)
   assert arrays[0]==arrays[2]and arrays[1]==arrays[3]
   idx=arrays[0][16:-16];adj=arrays[1][16:16+e];assert len(idx)==n+1 and idx[0]==0 and idx[-1]==e and all(a<=b for a,b in zip(idx,idx[1:]));assert all(0<=v<n for v in adj)
   graph=[set(adj[idx[v]:idx[v+1]])for v in range(n)];assert all(v in graph[w]for v in range(n)for w in graph[v]);labels=[-1]*n
   for start in range(n):
    if labels[start]!=-1:continue
    q=collections.deque([start]);labels[start]=start
    while q:
     v=q.popleft()
     for w in graph[v]:
      if labels[w]==-1:labels[w]=start;q.append(w)
   oracles.append(labels);checks.append(dict(n=n,e=e,components=len(set(labels)),complete_device_CSR_before_after_exact=True,undirected_graph_checked=True,independent_BFS_minimum_labels=True))
  assert offset==plan['input_prefix_count']
  for labels,c in zip(oracles,plan['cases']):
   n=c['n'];a=x[offset:offset+n+32];offset+=n+32;assert list(a[16:-16])==labels;assert all(v==plan['guard']for v in a[:16]+a[-16:]);assert all(v==plan['guard']for v in x[offset:offset+32]);offset+=32
  assert offset==len(x);records.append(dict(model=model,trial=t['trial'],artifact=dict(path=str(p.relative_to(R)),sha256=sha,bytes=len(raw)),full_values_checked=len(x),cases=checks,all_exact=True))
assert len(records)==20
(P/'saved_output_audit.json').write_text(json.dumps(dict(result=dict(path=str(F.relative_to(R)),sha256=hashlib.sha256(F.read_bytes()).hexdigest()),all20_pass=True,records=records,actual_source_bindings=bindings,unique_artifact_sha256=sorted(set(x['artifact']['sha256']for x in records)),independence='BFS reconstructed fromactualdeviceCSRbefore/afterreadbacks; noexpectedlabelassetloaded.',raw_captures_not_to_commit=True),indent=2)+'\n');print('CC_FULL20_INDEPENDENT_BFS_PASS',len(records),len(bindings))
