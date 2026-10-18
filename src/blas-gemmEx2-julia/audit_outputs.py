"""Full saved managed-buffer replay; independent integer matrix dot checks."""
from pathlib import Path
import hashlib,json,struct,sys,math
p=Path(__file__).resolve().parent;r=p.parents[1]
run=sys.argv[1];bundle=r/'.porting-state/portable_julia/runs'/run;d=json.loads((bundle/'result.json').read_text());fd=json.loads((p/'fixture_description.json').read_text());prefix=(p/'fixture.bin').read_bytes();gold=(p/'oracle.bin').read_bytes();kinds=(p/'kinds.bin').read_bytes();expected=struct.unpack('<'+str(len(gold)//8)+'d',gold);checks=[];hashes=set()
for model,m in d['models'].items():
 if not m['trials']:continue
 for t in m['trials']:
  f=Path(t['output_artifact']['saved_path']);raw=f.read_bytes();assert len(raw)==fd['artifact_count']*8;assert raw[:len(prefix)]==prefix;state=raw[len(prefix):];v=struct.unpack('<'+str(len(expected))+'d',state);assert all(math.isfinite(x)for x in v);assert all(x==y for x,y in zip(v,expected));assert all(state[i*8:(i+1)*8]==gold[i*8:(i+1)*8]for i,k in enumerate(kinds)if k)
  off=0;dots=0
  for case in fd['paths']:
   M,N,K=case['m'],case['n'],case['k'];la,lc=case['input_guard_elements'],case['output_guard_elements'];buf=[]
   for count,lead in [(M*N,lc),(M*N,lc),(M*K,la),(K*N,la)]:buf.append(v[off+lead:off+lead+count]);off+=count+2*lead
   initial,C,A,B=buf;scale=1 if case['dtype']==8 else 4
   for i in range(M):
    for j in range(N):
     a=[int(x*scale)for x in A[i*K:(i+1)*K]];b=[int(B[q*N+j]*scale)for q in range(K)];assert all(abs(x)<=2 for x in a+b);z=sum(x*y for x,y in zip(a,b));assert C[i*N+j]==z/(scale*scale);dots+=1
  assert len(v)-off==512 and all(x==165 for x in v[off:]);assert len(t['path_times_seconds'])==27;assert math.isclose(sum(t['path_times_seconds']),t['time_seconds'],rel_tol=1e-12,abs_tol=1e-15);hh=hashlib.sha256(raw).hexdigest();hashes.add(hh);checks.append(dict(model=model,trial=t['trial'],sha256=hh,full_words=len(v)+len(prefix)//8,dot_products=dots,actual_readonly_initialC_and_guards_bit_exact=True,full_outputs_exact_integer_oracle=True,workspace_guard_bytes_checked=512,path_clock_sum_pass=True))
 assert len(m['trials'])==5;assert math.isclose(m['mean_seconds'],sum(x['time_seconds']for x in m['trials'][1:])/4,rel_tol=1e-14)
out=dict(run_id=run,all_pass=True,models=len(d['models']),trials=len(checks),artifact_hash_classes=len(hashes),input_prefix_count=fd['input_prefix_count'],full_words=fd['artifact_count'],acceptance='Exact full finite arrays plus bit-exact all actual input/initialC/readonly/guards; recomputed Python integer dot from captured A/B; 27 path clocks sum, last4 mean independently recomputed.',max_abs_error=0,max_rel_error=None,relative_error_definition='Not computed; exact absolute equality holds including zero. No fabricated relative-zero oracle metric.',checks=checks)
(p/'output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='checks'}))
