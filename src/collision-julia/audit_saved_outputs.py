"""CPU-only independent exact full integer outputs and actual device-input echoes."""
from pathlib import Path
import json,struct,hashlib,sys
p=Path(__file__).resolve().parent;raw=(p/'fixture.bin').read_bytes();words=list(struct.unpack('<1282I',raw));expected=words.copy();cases=[]
for ci in range(40):
 a=list(struct.unpack('<32i',raw[ci*128:(ci+1)*128]));flag=len(set(a))<32;mask=0
 # Quadratic independent existential oracle, not GPU bitonic implementation.
 for i,x in enumerate(a):
  if any(y==x for y in a[:i]):mask|=1<<i
 assert mask.bit_count()==32-len(set(a))
 groups={}
 for i,x in enumerate(a):groups.setdefault(x,[]).append(i)
 assert {i for i in range(32) if not(mask>>i)&1}=={min(g) for g in groups.values()}
 if ci<32:assert mask==(((1<<32)-1)<<(32-ci))&0xffffffff if ci else mask==0
 expected.extend([int(flag)]*32+[mask]);expected.extend([(-991827)&0xffffffff]*8+words[ci*32:(ci+1)*32]+[(-991827)&0xffffffff]*8)
 cases.append({'case':ci,'unique_values':len(set(a)),'flag':flag,'expected_mask_hex':hex(mask),'mask_popcount':mask.bit_count(),'nonadjacent_control':any(x in a[:max(i-1,0)] and (i==0 or a[i-1]!=x) for i,x in enumerate(a))})
expectedbytes=struct.pack('<4522I',*expected);checks=[]
for arg in sys.argv[1:]:
 run=Path(arg);res=json.loads((run/'result.json').read_text())
 for model,md in res['models'].items():
  for t in md.get('trials',[]):
   path=Path(t['output_artifact']['saved_path']);got=path.read_bytes();assert got==expectedbytes,(model,t['trial']);assert len(t['path_times_seconds'])==40 and abs(sum(t['path_times_seconds'])-t['time_seconds'])<1e-12;checks.append({'model':model,'trial':t['trial'],'path':str(path),'sha256':hashlib.sha256(got).hexdigest(),'all_full_outputs_device_inputs_actual_inputguards_exact':True})
print(json.dumps({'prefix_words':1282,'artifact_words':4522,'flags_pertrial':1280,'masks_pertrial':40,'actual_device_input_echo_words':1920,'input_guards_pertrial':640,'output_guards_available':False,'native_output_layout':'sourcefixeddeviceglobals32flagsand1mask, noinventedoutputcanaries','cases':cases,'checks':checks},indent=2))
