from pathlib import Path
import struct,subprocess,json,time,hashlib,datetime,os,fcntl
r=Path('/home/imo/HeCBench_julia');p=r/'.porting-state/portable_julia/native_probes/floydwarshall2';p.mkdir(parents=True,exist_ok=True)
n=257;adj=[{} for _ in range(n)]
for u in range(n):
 for jump in (1,7,31,127):
  v=(u+jump)%n
  if v!=u:adj[u][v]=1+(u*17+jump*3)%37
idx=[0];nei=[];wt=[]
for a in adj:
 for v,w in sorted(a.items()):nei.append(v);wt.append(w)
 idx.append(len(nei))
graph=p/'named257.egr';graph.write_bytes(struct.pack('<'+'i'*(2+len(idx)+2*len(nei)),n,len(nei),*idx,*nei,*wt))
def run(name,cmd,env=None):
 start=time.time();q=subprocess.run(cmd,cwd=r/'src/floydwarshall2-cuda',env=env,capture_output=True,text=True,timeout=180);log=p/(name+'.log');log.write_text(q.stdout+q.stderr)
 rec={'command':cmd,'returncode':q.returncode,'wall_seconds':time.time()-start,'log':str(log.relative_to(r)),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),'original_source_sha256':hashlib.sha256((r/'src/floydwarshall2-cuda/main.cu').read_bytes()).hexdigest(),'scope':'Unmodified original CLI on named257-node synthetic directedpositive weightedCSR, repeat3; missing CollegeMsg default dataset not claimed.'};(p/(name+'.json')).write_text(json.dumps(rec,indent=2)+'\n');print(name,q.returncode,q.stdout[-1000:],q.stderr[-1000:],flush=True);return q
build=run('original_build01',['/usr/bin/nvcc','-O3','-std=c++17','-arch=sm_70','-I'+str(r/'src/mis-cuda'),'main.cu','-o',str(p/'original')]);assert build.returncode==0
with open('/tmp/hecbench-julia-study-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='6';q=run('original_execution01',[str(p/'original'),str(graph),'3'],env);assert q.returncode==0 and 'results match' in q.stdout and 'ERROR' not in q.stdout+q.stderr
