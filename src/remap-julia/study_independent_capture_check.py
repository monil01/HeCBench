"""Independent complete byte/oracle and internal-permutation review. No GPU execution."""
from pathlib import Path
import json,struct,hashlib,sys
p=Path(__file__).resolve().parent
read=lambda b:list(struct.unpack('<'+'i'*(len(b)//4),b))
f=read((p/'fixtures_i32.bin').read_bytes());cases=[];pos=1
for _ in range(f[0]):
 n=f[pos];pos+=1;cases.append(f[pos:pos+n]);pos+=n
expected=f.copy();sent=-123456789
for a in cases:
 s=sorted(a);u=sorted(set(a));rank={v:i for i,v in enumerate(u)}
 expected+=s+[len(u)]+u+[s.index(v) for v in u]+[rank[v] for v in a]+[sent]*512
expect_bytes=struct.pack('<'+'i'*len(expected),*expected)
rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());checks=[]
for model,data in r['models'].items():
 for trial in data.get('trials',[]):
  art=trial['output_artifact'];path=Path(art.get('saved_path',art['path']));path=path if path.exists() else rp.parent/path.name
  raw=path.read_bytes();assert raw==expect_bytes,(model,trial['trial'],'output bytes')
  perms=trial['permutation_witness'];assert len(perms)==len(cases)
  for a,permutation in zip(cases,perms):
   assert sorted(permutation)==list(range(len(a)))
   assert [a[j] for j in permutation]==sorted(a)
  stages=trial['stage_times_seconds'];assert len(stages)==4 and all(t>=0 for t in stages)
  assert abs(sum(stages)-trial['time_seconds'])<=1e-12
  checks.append({'model':model,'trial':trial['trial'],'full_bytes_exact':True,'complete_sort_permutation_bijection_and_membership':True,'stage_sum_correct':True,'sha256':hashlib.sha256(raw).hexdigest()})
print(json.dumps({'result':str(rp),'input_prefix_count':len(f),'count':len(expected),'all_pass':True,'checked_trials':checks},indent=2))
