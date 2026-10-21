"""Private text-only proposal. Never import/execute proposed method tools."""
from pathlib import Path
import json,hashlib,difflib,shutil
O=Path(__file__).parent;V4=O.parent/'v4';R=Path('/home/imo/HeCBench_julia');C=O.parent.parent/'profiler2025_calibration003';P=O/'proposed/tools';P.mkdir(parents=True,exist_ok=True);h=lambda b:hashlib.sha256(b).hexdigest()
inv=json.loads((V4/'delivery_inventory.json').read_text())
for f,digest in inv.items():
 if h((V4/f).read_bytes())!=digest:raise ValueError('V4drift '+f)
names=['v100_roofline_collect.py','v100_roofline_summarize.py','v100_roofline_audit.py','v100_profiler_method.py'];m={n:(V4/'proposed/tools'/n).read_text()for n in names}
controller_h=h((C/'calibrate_controller.py').read_bytes());manifest_h=h((C/'source_binding_manifest.json').read_bytes());g='v100_profiler_method.py'
m[g]=m[g].replace("CONTROLLER='7211c35d2d5fc02e157b1142622cc31c6fd78510685e3b8244f15f5d675caf26'","CONTROLLER='"+controller_h+"'").replace("MANIFEST='df47f5656c48ce113f126a92570a216fc50fe6352f939729e51a82e3a45241a0'","MANIFEST='"+manifest_h+"'").replace('002','003')
def sub(n,old,new):
 if m[n].count(old)!=1:raise ValueError('Ambiguous exact source edit '+n+' '+old[:80])
 m[n]=m[n].replace(old,new,1)
normalizer='''def normalize_calibrated_frequency_units(group):
 """Method-scoped factor1 unit view; raw CSV/group and numbers stay unchanged.
 Only the three exact elapsed-cycle frequency names accept raw 'hz'.
 No prefix, unknown unit, arbitrary per_second, or numeric conversion admitted.
 """
 eligible=('dram__cycles_elapsed.avg.per_second','sm__cycles_elapsed.avg.per_second','lts__cycles_elapsed.avg.per_second')
 view=dict(group);view['units']=dict(group['units']);aliases=[]
 for metric in eligible:
  need(metric in group['metrics'],'required calibrated frequency metric missing '+metric)
  unit=group['units'].get(metric)
  need(unit in ('cycle/second','hz'),'unsupported calibrated frequency unit '+metric+' '+str(unit))
  if unit=='hz':
   view['units'][metric]='cycle/second'
   aliases.append({'metric':metric,'raw_unit':'hz','validated_unit':'cycle/second','factor':1,'scope':'Explicit003 calibrated frequency dimension only; raw units and numeric values retained.'})
 return view,aliases

'''
sub(g,'def audit_profile(provenance,files):',normalizer+'def audit_profile(provenance,files):')
sub(g,"  need(all(k['scope_context'].get('profiler_method')==m and k['fp16_convention_evidence']==op and k['fp16_lane_factor']==2 for k in report['records']),'perkernel calibrationjoin mismatch')", """  need(all(k['scope_context'].get('profiler_method')==m and k['fp16_convention_evidence']==op and k['fp16_lane_factor']==2 for k in report['records']),'perkernel calibrationjoin mismatch')
  need(len(report['raw_metrics'])==len(report['records']),'raw/derived method row count mismatch')
  for group,kernel in zip(report['raw_metrics'],report['records']):
   _,aliases=normalize_calibrated_frequency_units(group)
   need(kernel['scope_context'].get('profiler_frequency_unit_aliases')==aliases,'perkernel exact frequency unitview evidence mismatch')""")
c='v100_roofline_collect.py'
sub(c,"                        records.append(parser.derive(group,half_lanes=2,", """                        derive_group=group
                        if method:
                            derive_group,unit_aliases=method_gate.normalize_calibrated_frequency_units(group)
                            row_context['profiler_frequency_unit_aliases']=unit_aliases
                        records.append(parser.derive(derive_group,half_lanes=2,""")
a='v100_roofline_audit.py'
sub(a,"                    for group, saved in zip(groups, report['records']):\n                        values = dict", """                    for group, saved in zip(groups, report['records']):
                        if prov.get('profiler_method'):
                            # Exact frozen method unit view, NOT a global parser alias.
                            group,unit_aliases=method_module.normalize_calibrated_frequency_units(group)
                            require(saved['scope_context'].get('profiler_frequency_unit_aliases')==unit_aliases,'Frozen frequency unitview evidence mismatch')
                        values = dict""")
for n,s in m.items():(P/n).write_text(s)
# Packer source archive checks use the NEW003 manifest, with unchanged old inputs.
shutil.copyfile(V4/'pack_method.py',O/'pack_method.py');(O/'calibration_input_sources').mkdir(exist_ok=True)
for f in (V4/'calibration_input_sources').iterdir():
 if f.is_file():shutil.copyfile(f,O/'calibration_input_sources'/f.name)
source_archive=json.loads((V4/'calibration_input_archive.json').read_text());source_archive['manifest_sha256']=manifest_h;(O/'calibration_input_archive.json').write_text(json.dumps(source_archive,indent=2)+'\n')
basepatch=[];delta=[];bindings=[]
for n in names:
 if n=='v100_profiler_method.py':base='';basepath='/dev/null'
 else:base=(R/'tools'/n).read_text();basepath='a/tools/'+n
 basepatch.extend(difflib.unified_diff(base.splitlines(True),m[n].splitlines(True),fromfile=basepath,tofile='b/tools/'+n));delta.extend(difflib.unified_diff((V4/'proposed/tools'/n).read_text().splitlines(True),m[n].splitlines(True),fromfile='v4/tools/'+n,tofile='v5/tools/'+n));bindings.append({'file':'tools/'+n,'base_sha256':h(base.encode())if base else None,'v4_sha256':h((V4/'proposed/tools'/n).read_bytes()),'v5_sha256':h((P/n).read_bytes())})
(O/'minimal.patch').write_text(''.join(basepatch));(O/'v4_to_v5.patch').write_text(''.join(delta));(O/'authoring_bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
# Body-preservation review, not execution of source functions.
bs=(R/'tools/v100_roofline_summarize.py').read_text();ba=(R/'tools/v100_roofline_audit.py').read_text();fragments={'summary_allreduce':bs[bs.index('        method_admission=None;method_hold=False'):bs.index('        runtime_review=')],'audit_allreduce':ba[ba.index("                if prov.get('allreduce_two_device_method'):"):ba.index("                if 'derived.json' in files:")],'selection_score':bs[bs.index('        score=('):bs.index('        key=(benchmark,model)')]}
for k,s in fragments.items():
 if s not in(m[a]if k.startswith('audit')else m['v100_roofline_summarize.py']):raise ValueError('Gate/score body changed '+k)
# V4F1 counter receipt and hold body remains literal in V5.
s4=(V4/'proposed/tools'/c).read_text();start=s4.index('            if method:\n                # run() returns');end=s4.index('            if isolated_review:',start);preserved=s4[start:end]
if preserved not in m[c]:raise ValueError('F1counter preservation changed')
review={'status':'source_only_root_review_ready_not_applied','controller003_sha256':controller_h,'manifest003_sha256':manifest_h,'calibration003_pass_claim':False,'GPU_build_tests_imports':0,'retained_F1_body_sha256':h(preserved.encode()),'allreduce_and_selection_fragments':{k:h(s.encode())for k,s in fragments.items()},'parser_live_unchanged_sha256':h((R/'tools/v100_roofline_classify.py').read_bytes()),'normalization':'Only exact3frequency unit names hz→cycle/second, numericfactor1 in a copied unit dictionary; rawgroups, CSV and all math formulas unchanged.','calibration002':'OriginalFAILED preserved; seven-row CPU diagnostic is not003pass or9rowadmission.'}
(O/'coexistence_source_review.json').write_text(json.dumps(review,indent=2)+'\n')
print(json.dumps({'new_controller':controller_h,'new_manifest':manifest_h,'private_v5':str(O),'no_proposed_execution':True}))
