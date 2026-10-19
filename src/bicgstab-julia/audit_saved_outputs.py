#!/usr/bin/env python3
"""Read-only full saved-capture audit; independent coordinate math, no GPU."""
import array,hashlib,json,math,pathlib,re,shutil,subprocess,sys,tempfile
P=pathlib.Path(__file__).resolve().parent;R=P.parent.parent
F=pathlib.Path(sys.argv[1]);F=F if F.is_absolute()else R/F;r=json.loads(F.read_text());n=1024;e=4992;cap=27617;assert r['reference_agreement']and r['matched_performance_valid']
rows=[0];cols=[];vals=[]
for i in range(n):
 js=sorted([i]+([i-32]if i>=32 else[])+([i-1]if i%32 else[])+([i+1]if i%32!=31 else[])+([i+32]if i<992 else[]));cols.extend(js);vals.extend(4.25 if j==i else -1. for j in js);rows.append(len(cols))
# Independent sparse row-elimination ILU0 in Python maps; no vendor reference.
factors=[dict(zip(cols[rows[i]:rows[i+1]],vals[rows[i]:rows[i+1]]))for i in range(n)]
for i,row in enumerate(factors):
 for j in sorted(k for k in row if k<i):
  row[j]/=factors[j][j]
  for k,v in factors[j].items():
   if k>j and k in row:row[k]-=row[j]*v
fac=[factors[i][j]for i in range(n)for j in cols[rows[i]:rows[i+1]]]
checked=[];bindings={};allsha=set();prefixsha=set();library_evidence=[]
for model,v in r['models'].items():
 assert v['complete']and v['provenance_valid']and v['timing_valid']and len(v['trials'])==5
 for path,sha in v['adapter_file_hashes'].items():
  assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==sha,path
  bindings[path]=sha
 for t in v['trials']:
  art=t['output_artifact'];p=pathlib.Path(art['saved_path']);b=p.read_bytes();sha=hashlib.sha256(b).hexdigest();assert sha==art['sha256']and len(b)==55236*8
  a=array.array('d');a.frombytes(b);assert all(map(math.isfinite,a));pre=a[:cap];post=a[cap:2*cap];off=n+1+3*e
  assert list(pre[:n+1])==rows and list(pre[n+1:n+1+e])==cols and list(pre[n+1+e:n+1+2*e])==vals
  ilu=max(abs(x-y)for x,y in zip(pre[n+1+2*e:off],fac));assert ilu<=1e-12
  assert pre[:off]==post[:off]
  def vec(z,k):return z[off+k*(n+32)+16:off+k*(n+32)+16+n]
  for z in [pre,post]:
   for k in range(11):
    start=off+k*(n+32);assert list(z[start:start+16])==[91.25]*16 and list(z[start+16+n:start+32+n])==[91.25]*16
  rhs=[.75*(4.25-(rows[i+1]-rows[i]-1))for i in range(n)];assert list(vec(pre,0))==rhs and vec(pre,0)==vec(post,0)
  assert all(x==0. for k in range(1,11)for x in vec(pre,k))
  x=vec(post,1);res=[rhs[i]-math.fsum(vals[k]*x[cols[k]]for k in range(rows[i],rows[i+1]))for i in range(n)];resnorm=math.sqrt(math.fsum(v*v for v in res));err=max(abs(v-.75)for v in x);resstored=max(abs(v-w)for v,w in zip(res,vec(post,3)))
  assert err<=2e-8 and resnorm<=2e-9 and resstored<=1e-12
  assert a[-2]==t['iterations']==77 and abs(a[-1]-resnorm)<=1e-12
  assert t['correct']and t['jit_excluded']and t['time_seconds']>0
  allsha.add(sha);prefixsha.add(hashlib.sha256(b[:cap*8]).hexdigest());checked.append(dict(model=model,trial=t['trial'],artifact=str(p.relative_to(R)),sha256=sha,count=len(a),exact_input_prefix_values=cap,solution_max_abs_error=err,true_residual_norm=resnorm,stored_residual_max_abs_disagreement=resstored,ILU0_max_abs_disagreement=ilu,iterations=t['iterations']))
 # Actual loaded CABI library evidence emitted by each process, not package guesses.
 source=r['run_id']if not v.get('reused_from_run_id')else v['reused_from_run_id'];log=R/'.porting-state/portable_julia/runs'/source/(model+'_run1.log')
 while not log.exists():
  parent=json.loads((log.parent/'result.json').read_text())['models'][model]['reused_from_run_id'];log=log.parent.parent/parent/(model+'_run1.log')
 for line in log.read_text().splitlines():
  if line.startswith('STUDY_LIBRARY {'):
   d=json.loads(line[len('STUDY_LIBRARY '):]);kind=d['kind'];path=pathlib.Path(d['path']);version=d['version']
  elif line.startswith('STUDY_LIBRARY kind='):
   m=re.fullmatch(r'STUDY_LIBRARY kind=(\d) path=(\S+) version=(\d+)',line);assert m;kind=['cublas','cusparse'][int(m[1])];path=pathlib.Path(m[2]);version=int(m[3])
  else:continue
  library_evidence.append(dict(model=model,kind=kind,path=str(path),realpath=str(path.resolve()),version=version,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),actual_log=str(log.relative_to(R))))
assert len(checked)==20 and len(allsha)==1 and len(prefixsha)==1 and len(library_evidence)==8
for kind in ['cublas','cusparse']:
 assert len({(v['version'],v['sha256'])for v in library_evidence if v['kind']==kind})==1
with tempfile.TemporaryDirectory(prefix='bicgstab-cold-cpu-')as tmp:
 q=pathlib.Path(tmp);(q/'src/bicgstab-julia').mkdir(parents=True);shutil.copytree(R/'src/bicgstab-cuda',q/'src/bicgstab-cuda');shutil.copy(P/'generate_bridge.py',q/'src/bicgstab-julia/generate_bridge.py');subprocess.run([sys.executable,'src/bicgstab-julia/generate_bridge.py'],cwd=q,check=True);subprocess.run([sys.executable,str(P/'native_diagnostics/generate_readiness_matrix.py'),'src/bicgstab-julia/study_grid.mtx'],cwd=q,check=True)
 cold=[]
 for name in ['native_solver.inc','study_bridge.cu','native_extraction.json','study_grid.mtx']:
  a=(P/name).read_bytes();b=(q/'src/bicgstab-julia'/name).read_bytes();assert a==b; cold.append(dict(path=name,bytes=len(a),sha256=hashlib.sha256(a).hexdigest(),byte_exact=True))
out=dict(benchmark='bicgstab',result=str(F.relative_to(R)),result_sha256=hashlib.sha256(F.read_bytes()).hexdigest(),status='pass',GPU_executions_added=0,all20_artifacts_bit_identical=True,artifact_count_each_trial=55236,unique_complete_artifact_sha256=sorted(allsha),exact_input_prefix_values=cap,actual_inputs_and_states_and_guards_fully_read=True,source_bindings=bindings,independent_checks='Coordinate5pointSPDknownsolution/residual+PythonmapILU0, fullfiniteworkspaces/guards, exactoriginalCSR/RHS/initialstate, all77iterationcounts. Not TEM application accuracy.',libraries=library_evidence,cold_reproduction=cold,trials=checked,mathematical_justification='Strict diagonal dominance margin0.25 => unique manufactured solution x=.75; ||error||inf <=||residual||inf/0.25. Source nonstandard P=R recurrence preserved; fixture convergence not general algorithm equivalence to standard BiCGStab.')
(P/'saved_output_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status='pass',trials=len(checked),bindings=len(bindings),cold_exact=len(cold),same_loaded_vendor_libraries=True,unique_complete_artifacts=len(allsha))))
