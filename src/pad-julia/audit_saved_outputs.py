"""CPU-only independent bytewise full output/flag/input proof for saved study bundles."""
from pathlib import Path
import struct,hashlib,json,sys
p=Path(__file__).resolve().parent
raw=(p/'fixture.bin').read_bytes();words=list(struct.unpack('<'+'Q'*(len(raw)//8),raw));gold=words.copy();partitions=[]
db=lambda x:struct.unpack('<Q',struct.pack('<d',x))[0]
for shape,(m,n,padding,offset) in enumerate([(256,127,1,0),(256,503,9,32768)]):
 size=m*(n+padding);nt=size//8192;assert size%8192==0
 # Input-scatter oracle independent from implementation destination gather.
 padded=[db(0.)]*size
 for src in range(m*n):
  row,col=divmod(src,n);padded[row*(n+padding)+col]=words[offset+src]
 for ai,alpha in enumerate([0.,.1,.25,.5,.75,1.]):
  af=struct.unpack('<f',struct.pack('<f',alpha))[0];cut=int(struct.unpack('<f',struct.pack('<f',nt*af))[0]);cap=(nt-cut)*8192;assert cap<=size
  paramoff=163840+(shape*6+ai)*9;assert words[paramoff:paramoff+9]==[m,n,padding,struct.unpack('<I',struct.pack('<f',af))[0],256,8,4,32,nt]
  gold.extend([db(-12345.)]*8+padded+[db(-12345.)]*8);gold.extend([db(-12345.)]*16)
  hf=[0]*(nt+1);hf[0]=hf[cut]=1
  for task in range(cut):hf[task+1]+=1
  gold.extend(hf)
  gf=[0]*(nt+1);gf[0]=gf[cut]=1
  for task in range(cut,nt):gf[task+1]+=1
  gold.extend([(-991827)&0xffffffff]*8+gf+[(-991827)&0xffffffff]*8)
  cpu=[t for worker in range(4) for t in range(worker,cut,4)];gpu=[t for block in range(8) for t in range(cut+block,nt,8)];assert sorted(cpu+gpu)==list(range(nt));assert set(cpu).isdisjoint(gpu)
  partitions.append({'shape':shape,'alpha_float32':af,'cpu_cut':cut,'tasks':nt,'actual_cpu_task_fraction':cut/nt,'gpu_payload_cells':cap,'cpu_tasks':sorted(cpu),'gpu_tasks':sorted(gpu)})
expected=struct.pack('<'+'Q'*len(gold),*gold);assert len(gold)==1147828
out={'expected_artifact_count':len(gold),'prefix_count':len(words),'checked_output_flag_guard_values':len(gold)-len(words),'expected_artifact_sha256':hashlib.sha256(expected).hexdigest(),'partitions':partitions,'artifacts':[]}
for arg in sys.argv[1:]:
 run=Path(arg);files=sorted(run.glob('*_trial*_output_trial*.bin'));assert files
 for file in files:
  got=file.read_bytes();assert got==expected,str(file);out['artifacts'].append({'path':str(file),'sha256':hashlib.sha256(got).hexdigest(),'bytewise_independent_oracle_pass':True})
print(json.dumps(out,indent=2))
