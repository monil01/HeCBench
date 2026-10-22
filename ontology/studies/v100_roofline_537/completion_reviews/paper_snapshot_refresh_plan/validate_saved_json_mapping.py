# Actual saved JSON audit and literal source-codebook extraction only; no proposed exporter import/execute.
from pathlib import Path
import ast,json,hashlib,collections,datetime
O=Path(__file__).resolve().parent;P=O/'failure_table_exporter_proposal';R=Path('/home/imo/HeCBench_julia');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def codebook(path):
 tree=ast.parse(Path(path).read_text());codes=[];group=set()
 for node in tree.body:
  if isinstance(node,ast.Assign):
   if any(isinstance(t,ast.Name)and t.id=='CODES'for t in node.targets):codes=ast.literal_eval(node.value)
   if any(isinstance(t,ast.Name)and t.id=='GROUP'for t in node.targets):group=ast.literal_eval(node.value)
  if isinstance(node,ast.Expr)and isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Attribute)and isinstance(node.value.func.value,ast.Name):
   call=node.value
   if call.func.value.id=='CODES'and call.func.attr=='extend':codes.extend(ast.literal_eval(call.args[0]))
   if call.func.value.id=='GROUP'and call.func.attr=='update':group.update(ast.literal_eval(call.args[0]))
 return codes,group
codes,group=codebook(P/'prepare_v100_failure_tables.py');oldcodes,oldgroup=codebook(P/'prepare_v100_failure_tables.base.py');known={k for k,v in codes};oldknown={k for k,v in oldcodes};assert len(known)==len(codes)
def mapped(c):
 k='build_or_helper_diagnostic'if c in group else c
 return k if k in known else'other_unmapped_outcome'
report=[]
for folder in [Path('/tmp/roofline-completion-20261021/measurement_failure_census/final_snapshot'),Path('/tmp/hecbench-roofline-v100-20261020/completion_20261021_v19/measurement_failure_census')]:
 s=json.loads((folder/'summary.json').read_bytes());a=json.loads((folder/'all_frozen_attempts.json').read_bytes())['records'];v=json.loads((folder/'selected_latest_slots.json').read_bytes())['records'];assert s['integrity']['valid']and not s['integrity']['issues'];unknown=[];views={};oldlost={}
 for view in ['selected','latest','all_frozen_attempts']:
  rows=a if view=='all_frozen_attempts'else[r for r in v if r['view']==view];per={};loss={}
  for m in ['cuda','cuda_jl','ka','jacc']:
   rr=[r for r in rows if r['model']==m];counts=collections.Counter(mapped(r['operational_category'])for r in rr);assert sum(counts.values())==len(rr)
   if view!='all_frozen_attempts':assert len(rr)==537
   lost=[r for r in rr if('build_or_helper_diagnostic'if r['operational_category']in oldgroup else r['operational_category'])not in oldknown]
   loss[m]=len(lost);per[m]={'records':len(rr),'displayed_category_sum':sum(counts[k]for k,label in codes),'counts':dict(counts)};assert per[m]['records']==per[m]['displayed_category_sum'];unknown.extend(r['operational_category']for r in rr if mapped(r['operational_category'])=='other_unmapped_outcome')
  views[view]=per;oldlost[view]=loss
 assert not unknown
 report.append({'folder':str(folder),'input_sha256':{n:sha(folder/n)for n in ['summary.json','all_frozen_attempts.json','selected_latest_slots.json']},'source_progress_updated_at':s['source_progress_updated_at'],'views':views,'unmapped_source_categories':unknown,'old_displayed_rows_omitted_outcome_counts':oldlost,'old_snapshot_counts_preserved_if_all_old_categories_covered':not any(sum(v.values())for v in oldlost.values())})
source=ast.parse((R/'tools/export_v100_measurement_failures.py').read_text());f=next(n for n in source.body if isinstance(n,ast.FunctionDef)and n.name=='classify');returns=sorted({n.value.value for n in ast.walk(f)if isinstance(n,ast.Return)and isinstance(n.value,ast.Constant)and isinstance(n.value.value,str)});assert all(mapped(c)!='other_unmapped_outcome'for c in returns)
receipt=R/'ontology/studies/v100_roofline_537/completion_reviews/runtime_only_applicability_five_root/root_adoption.json';adopt=json.loads(receipt.read_bytes());assert adopt['total_scoped_rows']==12 and adopt['old_rows_preserved_exactly']is True
out={'status':'actual_saved_JSON_mapping_checks_passed_not_exporter_execution','proposed_source_sha256':sha(P/'prepare_v100_failure_tables.py'),'literal_codebook_rows':len(codes),'all_current_classifier_constant_return_categories_mapped':returns,'actual_saved_snapshots':report,'new_applicability_authority':{'path':str(receipt),'sha256':sha(receipt),'source_reviewed_NA_rows':12,'historical_JSON_counts_not_backfilled':True},'old_explicit_caption':'21 October 2026, 18:38 UTC','historical_caption_preservation':'Exact original literal remains reproduced by explicit --snapshot-label; no proposed exporter was run. Old saved category counts independently preserved.','unknown_gate_source_review':'Unmapped raw outcome retained in detailed CSV/codebook/unknown_categories.json, compact counts cover it, publication_ready=false and nonzero exit after writing diagnostic packet. Unknown branch not runtime exercised here.','no_GPU_build_tests_or_proposed_import_execution':True}
(O/'saved_json_mapping_validation.json').write_text(json.dumps(out,indent=2)+'\n');print('snapshots',len(report),'categories',len(returns),'old_omissions',[d['old_displayed_rows_omitted_outcome_counts']for d in report]);print('NAauthority',12)
