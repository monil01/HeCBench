"""CPU-only exact minima/guards and independent raw per-block clock property audit."""
from pathlib import Path
import json,struct,hashlib,sys,math
p=Path(__file__).resolve().parent;data=(p/'fixture.bin').read_bytes();words=list(struct.unpack('<1042I',data));art=words.copy();mask32=(1<<32)-1;cases=[]
for fi in range(2):
 vals=struct.unpack('<512f',data[fi*2048:(fi+1)*2048]);minimum=sorted(vals)[0];gold=struct.unpack('<I',struct.pack('<f',minimum))[0]
 for blocks in [1,32,8192]:
  art.extend([struct.unpack('<I',struct.pack('<f',-12345.))[0]]*8+[gold]*blocks+[struct.unpack('<I',struct.pack('<f',-12345.))[0]]*8)
  art.extend([(-991827)&mask32]*16+[1]*blocks);cases.append((fi,blocks,gold))
expected=struct.pack('<34134I',*art);rows=[]
for arg in sys.argv[1:]:
 run=Path(arg);res=json.loads((run/'result.json').read_text())
 for model,md in res['models'].items():
  if not md.get('trials'):continue
  for t in md['trials']:
   canonical=Path(t['output_artifact']['saved_path']).read_bytes();assert canonical==expected,(model,t['trial'],'minima/guards/properties')
   path=Path(t['raw_timer_artifact']['path']);raw=path.read_bytes();timer=struct.unpack('<32996q',raw);off=0;pathrows=[]
   for fi,blocks,gold in cases:
    q=timer[off:off+2*blocks+16];off+=len(q);assert q[:8]==q[-8:]==(-991827,)*8
    ds=[(b-a)&((1<<64)-1) for a,b in zip(q[8:8+blocks],q[8+blocks:8+2*blocks])];assert all(v>=0 for v in q[8:-8]);assert all(0<d<(1<<31) for d in ds)
    pathrows.append({'fixture':fi,'blocks':blocks,'all_block_delta_properties_pass':True,'elapsed_cycles_min':min(ds),'elapsed_cycles_max':max(ds),'elapsed_cycles_mean':sum(ds)/blocks})
   assert off==len(timer);assert abs(sum(t['path_times_seconds'])-t['time_seconds'])<1e-12
   rows.append({'model':model,'trial':t['trial'],'canonical_sha256':hashlib.sha256(canonical).hexdigest(),'raw_clock_path':str(path.resolve()),'raw_clock_sha256':hashlib.sha256(raw).hexdigest(),'raw_clock_count':len(timer),'properties':pathrows})
print(json.dumps({'prefix_words':1042,'canonical_artifact_count':34134,'actual_minima_per_trial':16450,'canonicalized_timer_properties_per_trial':16450,'guard_words_per_trial':192,'raw_clocks_are_nondeterministic_auxiliary':True,'no_clock_equality_walltime_or_crossSM_assertion':True,'expected_canonical_sha256':hashlib.sha256(expected).hexdigest(),'rows':rows},indent=2))
