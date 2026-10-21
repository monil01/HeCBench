from pathlib import Path
P=Path(__file__).resolve().parent
p=P/'run_proposal.py';t=p.read_text().replace('import argparse,array,','import argparse,base64,array,')
t=t.replace('def validate_outputs(slot,plan,token):','def validate_outputs(slot,plan,token,snapshot):').replace("if sorted(x['trial']for x in trials)","if [x['trial']for x in trials]")
t=t.replace("env=rs[rank]['effective_environment'];physical=str(rank+5)","env=rs[rank]['effective_environment'];physical=str(rank+5)\n   if any(env[f'STUDY_EXPECTED_GPU{physical}_{key.upper()}']!=snapshot['devices'][physical][key]for key in('pci','uuid')):raise RuntimeError('registry_expected_identity_snapshot_join')")
t=t.replace('full=validate_outputs(slot,plan,token);','full=validate_outputs(slot,plan,token,snap);')
needle="  plan=expand(bundle['models'][m],slot);dump(slot/'plan.json',plan)"
replacement='''  plan=expand(bundle['models'][m],slot);dump(slot/'plan.json',plan)
  # Exact snapshots originate before any target process. Binaries are hash-bound
  # separately; source and review bytes are frozen losslessly without rewriting.
  verify(bundle['all_readonly_bindings']);snapshots=[]
  for name,digest in bundle['all_readonly_bindings'].items():
   path=Path(name)
   if path.suffix in('.py','.cu','.h','.cpp','.jl','.toml')or name in bundle['snapshot_review_paths']:
    raw=path.read_bytes()
    if len(raw)>64*1024*1024:raise RuntimeError('source_snapshot_finite64MiBcap')
    snapshots.append({'path':name,'sha256':digest,'bytes':len(raw),'base64':base64.b64encode(raw).decode('ascii')})
  dump(slot/'source_snapshots.json',{'schema_version':1,'created_before_target':True,'files':snapshots})
  (slot/'collector_source.py').write_bytes(Path(bundle['runner']['path']).read_bytes())
  (slot/'parser_source.py').write_bytes(Path(plan['counter_parser']['path']).read_bytes())
  (slot/'calibration_review.json').write_bytes(Path(bundle['calibration_review']['path']).read_bytes())
  dump(slot/'snapshot_bindings.json',{str(slot/name):sha(slot/name)for name in('source_snapshots.json','collector_source.py','parser_source.py','calibration_review.json')})'''
t=t.replace(needle,replacement)
t=t.replace("'source_bound_before':True,'fullchecks_required'", "'source_bound_before':True,'readonly_bindings_before':dict(bundle['all_readonly_bindings']),'snapshot_bindings_before':json.loads((slot/'snapshot_bindings.json').read_text()),'fullchecks_required'")
t=t.replace("verify(bundle['all_readonly_bindings']);verify(targets)","verify(bundle['all_readonly_bindings']);verify(targets);verify(prov['snapshot_bindings_before']);prov['readonly_bindings_after']=dict(bundle['all_readonly_bindings']);prov['snapshot_bindings_after']=dict(prov['snapshot_bindings_before'])")
p.write_text(t)
