"""Independent exact stable-filter/count/prefix/guard reconstruction, noGPU."""
from pathlib import Path
import struct,json,sys,hashlib
p=Path(__file__).resolve().parent;raw=(p/'fixtures_i32.bin').read_bytes();f=list(struct.unpack('<'+'i'*(len(raw)//4),raw));pos=1;expected=f.copy();cases=[]
for ci in range(f[0]):
 n,ai=f[pos:pos+2];pos+=2;a=f[pos:pos+n];pos+=n;tasks=(n+8191)//8192;alpha=struct.unpack('<f',struct.pack('<f',ai*.1))[0];cut=int(tasks*alpha);out=[x for x in a if x!=0];cf=[0]*(tasks+1);gf=[0]*(tasks+1);cf[0]=gf[0]=1;cf[cut]=gf[cut]=1
 for t in range(cut):cf[t+1]+=cf[t]+sum(x!=0 for x in a[t*8192:(t+1)*8192])
 for t in range(cut,tasks):gf[t+1]+=gf[t]+sum(x!=0 for x in a[t*8192:(t+1)*8192])
 expected +=[len(out)]+out+cf+gf+[-777]*8;cases.append({'n':n,'alpha':alpha,'cut':cut,'tasks':tasks,'retained':len(out)})
assert pos==len(f);gold=struct.pack('<'+'i'*len(expected),*expected);rp=Path(sys.argv[1]).resolve();r=json.loads(rp.read_text());checks=[]
for m,d in r['models'].items():
 for t in d.get('trials',[]):
  q=Path(t['output_artifact']['saved_path']);b=q.read_bytes();assert b==gold,(m,t['trial']);checks.append({'model':m,'trial':t['trial'],'all_input_compacted_values_counts_CPU_GPU_flags_guards_exact_bytes':True,'sha256':hashlib.sha256(b).hexdigest()})
print(json.dumps({'result':str(rp),'count':len(expected),'prefix':len(f),'all_pass':True,'golden_sha256':hashlib.sha256(gold).hexdigest(),'cases':cases,'checks':checks},indent=2))
