"""Read-only source-fragment inventory; no imports/execution of proposed code."""
from pathlib import Path
import hashlib,json
W=Path('/home/imo/HeCBench_julia');O=Path(__file__).parent
s=(W/'tools/v100_roofline_summarize.py').read_text();ps=(O/'proposed/tools/v100_roofline_summarize.py').read_text();a=(W/'tools/v100_roofline_audit.py').read_text();pa=(O/'proposed/tools/v100_roofline_audit.py').read_text()
frags={'summarize_allreduce_admission':s[s.index('        method_admission=None;method_hold=False'):s.index('        runtime_review=')],'audit_allreduce_admission':a[a.index("                if prov.get('allreduce_two_device_method'):"):a.index("                if 'derived.json' in files:")],'unchanged_selection_score':s[s.index('        score=('):s.index('        key=(benchmark,model)')]}
rows=[]
for n,v in frags.items():
 revised=ps if n!='audit_allreduce_admission'else pa
 assert v in revised,n
 rows.append({'fragment':n,'sha256':hashlib.sha256(v.encode()).hexdigest(),'verbatim_present_in_v3':True,'qualification':'Source-only exact-byte comparison, not an executed behavioral regression test'})
pure=Path('/tmp/roofline-completion-20261021/profiler2025_calibration002/calibrate_controller.py');t=pure.read_text();binds=[]
for p in [W/'tools/v100_roofline_allreduce_admission.py',W/'ontology/studies/v100_roofline_537/completion_reviews/allreduce_two_device/root_cpu_review.json',pure,Path('/tmp/roofline-completion-20261021/profiler2025_calibration002/source_binding_manifest.json')]:binds.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(O/'coexistence_source_review.json').write_text(json.dumps({'allreduce_fragments':rows,'readonly_bindings':binds,'axes':'allreduce_two_device_method is independent from profiler_method. Both restrictions must pass if present; existing allreduce remains its reviewed2022-only method. No approval for2025allreduce is added.','math_and_method_gate_scope':'Classifier math and WMMAtypedtargetloader unchanged; original selection score body exact. Only newmethod admission/labels inserted.','code_imported_or_executed':False,'new_calibration_pass_or_adoption':False},indent=2)+'\n')
history={}
for name in ['delivery_inventory.json','minimal.patch']:
 for version in ['v1','v2']:
  p=O.parent/name if version=='v1'else O.parent/'v2'/name;history[version+'/'+name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(O/'revision_history.json').write_text(json.dumps({'preserved_previous_candidates':history,'reason':'Bind002controller7211/manifestdf47 and rebase independently restrictedallreduce branches505d/28faa','calibration_waiting_no_pass_claim':True},indent=2)+'\n');print(json.dumps({'fragments':len(rows),'all_verbatim':True,'no_shared_changes':True}))
