"""CPU-only audit of all saved field, matrix, index, guard and timing evidence."""
from pathlib import Path
import json,struct,hashlib,sys,math
p=Path(__file__).resolve().parent;meta=json.loads((p/'fixture_description.json').read_text());raw=(p/'fixture.bin').read_bytes();oracle=(p/'oracle.bin').read_bytes();prefix=struct.unpack('<%sd'%(len(raw)//8),raw);gold=struct.unpack('<%sd'%(len(oracle)//8),oracle);rows=[]
for arg in sys.argv[1:]:
 run=Path(arg);res=json.loads((run/'result.json').read_text());assert res['reference_agreement']
 for model,d in res['models'].items():
  assert len(d['trials'])==5 and all(t['correct'] for t in d['trials']);assert abs(d['mean_seconds']-sum(t['time_seconds'] for t in d['trials'][1:])/4)<1e-12
  for t in d['trials']:
   f=Path(t['output_artifact']['saved_path']);data=f.read_bytes();assert data[:len(raw)]==raw;got=struct.unpack('<%sd'%len(gold),data[len(raw):]);off=0;guards=0;tails=0
   for ci,c in enumerate(meta['cases']):
    n=c['n'];e=n//2
    for length,g,active in [(n*6,6,e*6),(n*72,18,e*72),(n*72,18,e*72)]:
     z=got[off:off+length+g*2];ref=gold[off:off+length+g*2];off+=length+g*2
     assert z[:g]==ref[:g] and z[-g:]==ref[-g:];guards+=2*g
     assert z[g+active:g+length]==ref[g+active:g+length];tails+=length-active
     assert all(math.isfinite(x) and abs(x-y)<=2e-5+2e-6*abs(y) for x,y in zip(z,ref))
     if ci==2 and length==n*6:assert all(x==0 for x in z[g:g+active])
    # Actualdevice payload and physical input guards must be exact, not tolerancechecked.
    for length,g in [(n*6,6),(n*72,18),(n*72,18)]+[(n*4,4)]*4:
     assert got[off:off+length+2*g]==gold[off:off+length+2*g];off+=length+2*g;guards+=2*g
   assert off==len(got);assert guards==600 and tails==135600
   assert all(float(struct.unpack('<f',struct.pack('<f',x))[0])==x for x in got) #UInt64 maps selectedsmall exactF32 too.
   assert abs(sum(t['path_times_seconds'])-t['time_seconds'])<1e-12
   for name,h in d.get('adapter_file_hashes',{}).items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,(model,name)
   rows.append(dict(model=model,trial=t['trial'],artifact=str(f),sha256=hashlib.sha256(data).hexdigest(),input_prefix_bytes_exact=True,all_output_values_equal_CPU_oracle_exact_secondary_selected_dyadic_check=got==gold,physical_guards_exact=guards,odd_unwritten_tail_values_exact=tails,identity_constant_hopping_output_zero=True,dependencies_current=True))
print(json.dumps(dict(actual_GPU_compute='Float32 complex matrix-vector arithmetic; UInt64 neighbor maps; Float64 artifact losslesswidening',artifact_values=872062,input_prefix_values=300134,prefix_bytes=len(raw),active_output_complex_components_real_values=5424,active_backward_matrix_real_values=130176,physical_guards_pertrial=600,untouched_odd_tail_values_pertrial=135600,independent_oracle='Explicit coordinate-neighbor complex3x3 matvec and conjugate transpose; PythonFloat64 exactdyadic data yields exactsecondaryagreement forselectedclass only',atol=2e-5,rtol=2e-6,rows=rows),indent=2))
