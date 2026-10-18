from pathlib import Path
import json,struct,hashlib,sys
p=Path(__file__).resolve().parent;m=json.loads((p/'fixture_description.json').read_text());unpack=lambda f:list(struct.unpack('<'+'i'*(Path(f).stat().st_size//4),Path(f).read_bytes()));prefix=unpack(p/'fixture.bin');gold=unpack(p/'oracle.bin');rows=[]
for arg in sys.argv[1:]:
 b=Path(arg);d=json.loads((b/'result.json').read_text())
 for model,v in d['models'].items():
  for t in v.get('trials',[]):
   f=Path(t['output_artifact']['saved_path']);a=unpack(f);assert len(a)==m['artifact_values'];assert a[:len(prefix)]==prefix;assert a[len(prefix):]==gold;pos=len(prefix);costs=[]
   for ci,c in enumerate(m['cases']):
    cs=a[pos:pos+256];rs=a[pos+256:pos+512];total=a[pos+512];cost=a[pos+513:pos+66049];pos+=66049
    assert sorted(cs)==list(range(256)) and sorted(rs)==list(range(256));assert all(rs[cs[r]]==r for r in range(256));assert cost==prefix[2+ci*65536:2+(ci+1)*65536];assert sum(cost[cs[r]*256+r] for r in range(256))==total==c['optimal_cost'];assert cs==c['expected_assignment'];assert c['unique_optimum_gap']>0;costs.append(total)
   assert pos==len(a);counts=t['stage_launch_counts'];assert len(counts)==48 and all(x>=0 for x in counts)
   for ci,c in enumerate(m['cases']):
    if c['requires_step6']:assert counts[ci*16+13]>0 and counts[ci*16+14]>0 and counts[ci*16+15]>0
   auxchecks=[]
   for z in t.get('auxiliary_artifacts',[]):
    af=Path(z['path'])
    if af.exists() and z.get('sha256'):
     assert af.stat().st_size==z['bytes'];assert hashlib.sha256(af.read_bytes()).hexdigest()==z['sha256'];raw=unpack(af);assert len(raw)==198144
     for ci in range(3):
      sl=raw[ci*66048:ci*66048+65536];cs=raw[ci*66048+65536:ci*66048+65792];rs=raw[ci*66048+65792:(ci+1)*66048];assert min(sl)>=0;assert cs==m['cases'][ci]['expected_assignment'];assert all(sl[cs[r]*256+r]==0 and rs[cs[r]]==r for r in range(256))
     auxchecks.append(dict(path=str(af),sha256=z['sha256'],diagnostic_nonnegative_slack_assigned_zero_pass=True))
   rows.append(dict(model=model,trial=t['trial'],all_checks_pass=True,exact_output_values=1539,exact_cost_echo_values=196608,input_prefix_values=len(prefix),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),optimal_costs=costs,stage_launch_counts=counts,auxiliary_states=auxchecks,auxiliary_missing_original_native=model=='cuda' and not auxchecks))
assert rows
print(json.dumps(dict(rows=rows,all_pass=all(x['all_checks_pass'] for x in rows),oracle='Independent exhaustive720permutations/core and offblockdominance/planted fullpermutation proof, unique optimum and allrawassignment/inverse/cost exact; not merely totalcost property',native_guard_scope='Original staticglobals have nophysicaloutputguards; no claim they exist. Juliaextra13buffer16guardwords checkedprogramside.',old_native_auxiliary_gap='Initial5native tempslack auxiliary files were not capturedbyrunner; not reconstructed. Primaryfullassignments/inputs preserved and valid.',native_input_echo_scope='Originalhostcostcopy serialized; sourcehash/cudaMemcpyToSymbol exactinput transfer binding, not a saved preexecutiondeviceinputreadback. Juliaactualpreexecutiondevicecopy independentlychecked.'),indent=2))
