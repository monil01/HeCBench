from pathlib import Path
import struct, json, hashlib
R=Path(__file__).resolve().parents[2]
p=R/'.porting-state/portable_julia/gerbil_diagnostics/native_probe_02_full_u32.bin'
raw=p.read_bytes();v=struct.unpack('<'+'I'*(len(raw)//4),raw)
nt=1024*3+16;nk=257*2+16
arrays=[v[:nt],v[nt:nt+nk],v[nt+nk:nt+nk*2]];table=arrays[0]
nc=struct.unpack('<3Q',raw[-24:]);rows={};duplicate=False
for i in range(1024):
 off=8+i*3
 if table[off+2]:
  key=table[off] | table[off+1]<<32
  duplicate |= key in rows
  rows[key]=rows.get(key,0)+table[off+2]
e=json.loads((R/'src/gerbil-julia/probe_manifest.json').read_text())
expected={int(k):n for k,n in e['counts']}
guards=all(z==0xa5a5a5a5 for array in arrays for z in array[:8]+array[-8:]) and nc[0]==nc[2]==0xa5a5a5a5a5a5a5a5
out={'source_run':'20261019T194955740158Z_gerbil','artifact':str(p.relative_to(R)),'artifact_sha256':hashlib.sha256(raw).hexdigest(),'artifact_bytes':len(raw),'full_complete_histogram_exact':rows==expected,'duplicate_table_keys':duplicate,'allguards_exact':guards,'noSuccess_counter':nc[1],'source_KMer_layout_witness':'src/gerbil-julia/source_key_layout_witness.log','scope':'One original32thread cooperative insert process check, not five-trial accepted port. Full CLI and targets unverified; synchronization concern is not an observed defect.'}
assert rows==expected and guards and nc[1]==0 and not duplicate
(R/'src/gerbil-julia/native_probe_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
