"""CPU-only full corrected-frame arrays, inputbits, guards, oracle and timing audit."""
from pathlib import Path
import struct,json,hashlib,math,sys
p=Path(__file__).resolve().parent;raw=(p/'fixture.bin').read_bytes();fix=struct.unpack('<35274d',raw);gold=struct.unpack('<100032d',(p/'oracle.bin').read_bytes());meta=json.loads((p/'fixture_description.json').read_text());echo=tuple(fix[6:970])+(-777.,)*16+tuple(fix[970:1482])+(-777.,)*16+(-777.,)*8+tuple(fix[1482:2506])+(-777.,)*8+(42405.,)*8+tuple(fix[2506:])+(42405.,)*8
assert len(echo)==35332
for row in range(128):assert sorted(struct.unpack_from('<256H',(p/'permutations.bin').read_bytes(),row*512))==list(range(256))
assert hashlib.sha256((p/'corrected_native/iteration.cu').read_bytes()).hexdigest()==hashlib.sha256((p/'original_native/iteration.cu').read_bytes()).hexdigest();rows=[];model_first={}
for arg in sys.argv[1:]:
 run=Path(arg);res=json.loads((run/'result.json').read_text());assert res['reference_agreement']
 for model,d in res['models'].items():
  assert len(d['trials'])==5 and all(t['correct'] for t in d['trials']);assert abs(d['mean_seconds']-sum(t['time_seconds'] for t in d['trials'][1:])/4)<1e-12
  for t in d['trials']:
   f=Path(t['output_artifact']['saved_path']);data=f.read_bytes();assert data[:len(raw)]==raw;v=struct.unpack('<135364d',data[len(raw):]);out=v[:100032];assert struct.pack('<35332d',*v[100032:])==struct.pack('<35332d',*echo);assert all(map(math.isfinite,out));off=0;guards=0;metrics=[]
   for case in range(2):
    maximum=0.;norm=0.;state_exact=True;colors_exact=True;linear_exact=True
    for kind,n,components in [('points',256,2),('colors',256,1),('vertices',16384,3)]:
     length=(n+16)*components;a=out[off:off+length];b=gold[off:off+length];off+=length
     assert a[:8*components]==b[:8*components] and a[-8*components:]==b[-8*components:];guards+=16*components
     for j,(x,y) in enumerate(zip(a,b)):
      guard=j<8*components or j>=(8+n)*components;color=kind=='colors' or kind=='vertices' and j%3==2
      if kind=='points':assert struct.unpack('<e',struct.pack('<e',x))[0]==x
      if kind=='colors':assert struct.unpack('<e',struct.pack('<e',x))[0]==x
      if kind=='vertices':assert struct.unpack('<f',struct.pack('<f',x))[0]==x
      if guard or color or case==1:assert x==y
      else:
       e=abs(x-y);maximum=max(maximum,e);q=e/(2**-9+2**-9*abs(y));norm=max(norm,q);assert q<=1
      if kind=='points':state_exact &= x==y
    metrics.append(dict(case=case,max_position_abs_error=maximum,max_engineering_budget_fraction=norm,half_state_points_exact_vs_independent_oracle=state_exact,all_colors_exact=True,linear_control_all_coordinates_exact=True if case==1 else None))
   assert off==100032 and guards==192;assert t['exact_color_guard_linear_mismatches']==0;assert abs(max(q['max_position_abs_error'] for q in metrics)-t['max_abs_error'])<1e-12;assert abs(sum(t['path_times_seconds'])-t['time_seconds'])<1e-12
   for name,h in d.get('adapter_file_hashes',{}).items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,(model,name)
   model_first.setdefault(model,data);assert data==model_first[model]
   rows.append(dict(model=model,trial=t['trial'],artifact=str(f),sha256=hashlib.sha256(data).hexdigest(),prefix_actual_GPUparam_and_input_bytes_exact=True,full_input_echo_and64guards_exact=True,output_physical_guards_exact=192,all_declared_dependencies_current=True,cases=metrics))
comparisons=[]
for model,blob in model_first.items():
 for ref in ['cuda','cuda_jl']:
  if ref not in model_first:continue
  a=struct.unpack('<170638d',blob);b=struct.unpack('<170638d',model_first[ref]);comparisons.append(dict(model=model,reference=ref,max_full_artifact_absolute_difference=max(abs(x-y) for x,y in zip(a,b)),raw_bitwise_equal=blob==model_first[ref]))
print(json.dumps(dict(artifact_values=170638,input_prefix_values=35274,input_prefix_bytes=len(raw),actual_state_scalar_values=1536,actual_vertex_scalar_values=98304,physical_guard_values_pertrial=256,byvalue_params_bytes=1928,configuration_role='corrected_reference_secondary',native_reference_repaired=True,oracle_precision_qualification=meta['oracle'],engineering_budget_basis=meta['contract_basis'],domain_summary=meta['state_domain'],no_dynamic_FLOP_or_renderer_imagequality_claim=True,comparisons=comparisons,rows=rows),indent=2))
