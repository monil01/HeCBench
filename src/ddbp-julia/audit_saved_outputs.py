"""Read-only full-stage/prefix/guard/source audit; does not execute benchmarks."""
from pathlib import Path
import hashlib,json,struct,sys,math
R=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
rp=Path(sys.argv[1]) if len(sys.argv)>1 else R/'.porting-state/portable_julia/runs/20261018T052043594655Z_ddbp/result.json'
r=json.loads(rp.read_text());meta=json.loads((P/'fixture_provenance.json').read_text());prefix=sum([x['prefix'] for x in meta['cases']],[]);prefixbytes=struct.pack('<360d',*prefix);oracle=[]
for ci in [1,2]:
 raw=(P/f'oracle_case{ci}_f64.bin').read_bytes();oracle+=list(struct.unpack('<'+'d'*(len(raw)//8),raw))
assert len(oracle)==2867
records=[];sources=[]
assert r['reference_agreement'] and all(r['variant_verified'].values()) and r['matched_performance_valid']
for m,v in r['models'].items():
 assert len(v['trials'])==5 and v['provenance_valid'] and v['timing_valid']
 for t in v['trials']:
  a=t['output_artifact'];f=Path(a['saved_path']);raw=f.read_bytes();assert len(raw)==4059*8 and hashlib.sha256(raw).hexdigest()==a['sha256'];assert raw[:360*8]==prefixbytes
  values=struct.unpack('<4059d',raw);states=values[360:3227];assert all(map(math.isfinite,values));assert raw[3227*8:]==struct.pack('<832d',*([-8192.]*832))
  assert all(abs(x-y)<=1e-10+1e-9*abs(y) for x,y in zip(states,oracle));stage_records=[];offset=0
  for case in meta['cases']:
   for label in case['state_labels']:
    start=offset+label['start'];count=label['count'];errs=[abs(states[i]-oracle[i]) for i in range(start,start+count)];stage_records.append(dict(case=case['case'],stage=label['label'],count=count,max_abs_error=max(errs)))
   offset+=case['oracle_count']
  records.append(dict(model=m,trial=t['trial'],artifact=str(f.relative_to(R)),sha256=a['sha256'],prefix_exact_values=360,guards_exact_values=832,complete_intermediate_values=2867,independent_max_abs_error=max(abs(x-y) for x,y in zip(states,oracle)),stages=stage_records))
 for filename,h in v['adapter_file_hashes'].items():
  f=Path(filename);assert hashlib.sha256(f.read_bytes()).hexdigest()==h;sources.append(dict(model=m,path=filename,sha256=h))
result=dict(result=str(rp.relative_to(R)),all20_complete_stage_artifacts_pass=True,records=records,measured_source_bindings=sources,reference_role='corrected_reference_secondary',original_primary_status='reference_blocked',original_failure_witness='native_quality_witness.json',oracle='Globalinclusiveaxisintegrals,parametricray/planeintersection,tensorproductbilinearweights and sqrtangleprefactors; unit-planeanalyticcontrol',guard_and_input_checks='Exactserializedbytesseparatefromfloatpayloadtolerance',limits=['Nooriginalracyprimarynumericalagreementclaimed.','DiagnosticrerunstatecapturesareoutsideROI; sameimmutableinputs/stagecode, nosilenttimedDtoH.','No medicalapplicationqualityclaim.'])
(P/'saved_output_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(artifacts=len(records),source_bindings=len(sources),max_abs_error=max(z['independent_max_abs_error'] for z in records))))
