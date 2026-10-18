"""CPU-only full typed input, grid-stride coverage, algebraic oracle and saved-output audit."""
from pathlib import Path
import json,struct,hashlib,sys,math
p=Path(__file__).resolve().parent;raw=(p/'fixture.bin').read_bytes();fix=struct.unpack('<294700d',raw);gold=struct.unpack('<294896d',(p/'oracle.bin').read_bytes());meta=json.loads((p/'fixture_description.json').read_text());pos=0;expected=[];echo=[];coverage=[]
for c in meta['cases']:
 ty,n,t,b,h=map(int,fix[pos:pos+5]);pos+=5;lanes=c['lanes'];a=fix[pos:pos+n*lanes];pos+=n*lanes
 assert all(x in [-1,0] for x in a);assert t==1 or all(x==-1 for x in a)
 expected += [-777.]*(8*lanes)+[-1. if x==-1 else -2. for x in a]+[-777.]*(8*lanes)
 echo += [-777.]*(8*lanes)+list(a)+[-777.]*(8*lanes)
 fmt='<e' if ty==1 else '<f' if ty==2 else '<d'
 assert all(struct.pack('<d',struct.unpack(fmt,struct.pack(fmt,x))[0])==struct.pack('<d',x) for x in a)
 total=b*h;seen=[]
 for start in range(total):seen += list(range(min(start,n),min(start+math.ceil(n/total)*total,n),total))
 assert sorted(seen)==list(range(n));coverage.append(dict(type=ty,n=n,blocks=b,threads=h,indices_each_covered_once=True,inner_trials=t,lanes=lanes))
assert pos==len(fix) and tuple(expected)==gold
packed_echo=struct.pack('<%sd'%len(echo),*echo);rows=[];first=None
for arg in sys.argv[1:]:
 run=Path(arg);res=json.loads((run/'result.json').read_text());assert res['reference_agreement']
 for model,d in res['models'].items():
  assert len(d['trials'])==5 and all(t['correct'] for t in d['trials']);assert abs(d['mean_seconds']-sum(t['time_seconds'] for t in d['trials'][1:])/4)<1e-12
  for t in d['trials']:
   f=Path(t['output_artifact']['saved_path']);data=f.read_bytes();first=first or data;assert data==first and data[:len(raw)]==raw
   out=struct.unpack('<294896d',data[len(raw):len(raw)+len(gold)*8]);assert out==gold;assert data[len(raw)+len(gold)*8:]==packed_echo
   ma=max(abs(a-b) for a,b in zip(out,gold));mr=max(abs(a-b)/abs(b) for a,b in zip(out,gold));assert ma==mr==0
   guards=0;off=0
   for c in meta['cases']:
    n,l=c['n'],c['lanes'];z=out[off:off+(n+16)*l];off+=(n+16)*l;assert z[:8*l]==(-777.,)*(8*l) and z[-8*l:]==(-777.,)*(8*l);guards+=16*l
   assert guards==256 # plus256preinputecho guards=512perartifact
   assert abs(sum(t['path_times_seconds'])-t['time_seconds'])<1e-12
   for name,h in d.get('adapter_file_hashes',{}).items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,(model,name)
   rows.append(dict(model=model,trial=t['trial'],artifact=str(f),sha256=hashlib.sha256(data).hexdigest(),max_abs_error_full_algebraic_oracle=ma,max_rel_error_full_nonzero_algebraic_oracle=mr,input_prefix_signedzero_bytes_exact=True,prelaunch_actual_input_echo_signedbits_exact=True,output_physical_guards_exact=256,input_echo_physical_guards_exact=256,all_declared_dependencies_current=True))
print(json.dumps(dict(artifact_values=884492,input_prefix_values=294700,input_prefix_bytes=len(raw),output_scalar_values=294640,actual_GPU_types=['packed2xFloat16','Float32','Float64'],artifact_type='Float64losslessencoding_selectedtypedvaluesincludinginputnegativezero',oracle='Independentexactfixedpointandzero-responsealgebra, fulloraclevaluesallnonzero so maxrelative0well-defined; no inference of executedFLOPs from invariantoutputs',physical_guards_pertrial=512,grid_stride_coverage=coverage,rows=rows),indent=2))
