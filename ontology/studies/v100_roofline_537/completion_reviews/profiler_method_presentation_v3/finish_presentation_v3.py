from pathlib import Path
import re,json,hashlib,difflib,datetime
R=Path('/home/imo/HeCBench_julia');O=Path('/tmp/roofline-completion-20261021/profiler_method_presentation_v3');P=Path('/tmp/roofline-completion-20261021/profiler_method_presentation_v2')
# Combine adjacent independent argument additions.
p=O/'proposed/tools/v100_roofline_refresh.py';s=p.read_text()
s=re.sub(r'^<<<<<<<[^\n]*\n(.*?)^=======\n(.*?)^>>>>>>>[^\n]*\n',lambda m:m[1]+m[2],s,flags=re.M|re.S);p.write_text(s)
# Keep actual live panel including display provenance and add distinct method identity fields.
p=O/'proposed/ontology/tools/julia_graph/graph.js';s=p.read_text()
def panel(m):
 live=m[2];anchor="+detail([['V100 profile status',m.roofline?.profiling_status]";assert anchor in live
 return live.replace(anchor,"+detail([['V100 profiler method',m.roofline?.profiler_method_id],['V100 profiler version',m.roofline?.profiler_version],['V100 method review SHA',m.roofline?.method_review_sha256],['V100 profile status',m.roofline?.profiling_status]",1)
s=re.sub(r'^<<<<<<<[^\n]*\n(.*?)^=======\n(.*?)^>>>>>>>[^\n]*\n',panel,s,flags=re.M|re.S);p.write_text(s)
# Snapshot unchanged consumer/authority implementation to bind preservation claims.
for rel in ['ontology/tools/current537_display_traits.py','ontology/tools/report/explorer.js']:
 for prefix in ['base','proposed']:
  dst=O/prefix/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((R/rel).read_bytes())
files=sorted(str(p.relative_to(O/'base'))for p in(O/'base').rglob('*')if p.is_file());rows=[];patch=[]
sha=lambda b:hashlib.sha256(b).hexdigest()
for rel in files:
 b=(O/'base'/rel).read_bytes();q=(O/'proposed'/rel).read_bytes();assert b'<<<<<<<'not in q and b'>>>>>>>'not in q
 rows.append({'path':rel,'base_sha256':sha(b),'proposed_sha256':sha(q),'live_sha256_at_finish':sha((R/rel).read_bytes()),'base_still_current':b==(R/rel).read_bytes(),'changed':b!=q})
 patch+=difflib.unified_diff(b.decode().splitlines(True),q.decode().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel)
(O/'minimal.patch').write_text(''.join(patch));(O/'source_inventory.json').write_text(json.dumps(rows,indent=2)+'\n')
# Text invariants only; no imports, compilation, application or tests.
b=(O/'base/ontology/tools/generate_relationship_graph.py').read_text();q=(O/'proposed/ontology/tools/generate_relationship_graph.py').read_text()
method_delta=''.join(difflib.unified_diff(b.splitlines(True),q.splitlines(True)))
refresh=(O/'proposed/tools/v100_roofline_refresh.py').read_text();prior=Path('/tmp/roofline-completion-20261021/current537_display_refresh_passthrough/future_method_and_traits_FD/tools/v100_roofline_refresh.py').read_text()
(O/'static_source_review.json').write_text(json.dumps({'scope':'Text comparison only; target code not run','graph_delta':method_delta,'edge_generation_from_candidates_onward_byte_identical':b[b.index('    candidates='):]==q[q.index('    candidates='):] if '    candidates='in b else None,'borrowed_FD_loader_byte_identical':(O/'base/ontology/tools/current537_display_traits.py').read_bytes()==(O/'proposed/ontology/tools/current537_display_traits.py').read_bytes(),'scientific_explorer_byte_identical':(O/'base/ontology/tools/report/explorer.js').read_bytes()==(O/'proposed/ontology/tools/report/explorer.js').read_bytes(),'refresh_equals_previous_composed_bytes':refresh==prior,'refresh_composed_difference':''.join(difflib.unified_diff(prior.splitlines(True),refresh.splitlines(True)))},indent=2)+'\n')
print(json.dumps(rows,indent=2))
