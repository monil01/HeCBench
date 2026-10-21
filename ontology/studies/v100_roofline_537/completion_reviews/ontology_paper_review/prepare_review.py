import json,hashlib,datetime,collections,csv
from pathlib import Path
R=Path('/home/imo/HeCBench_julia');O=Path('/tmp/roofline-completion-20261021/ontology_paper_review');S=R/'ontology/studies/v100_roofline_537'
files=[R/'2026-PASC-Hecbench-Julia-paper/sections/06_performance_roofline.tex',R/'2026-PASC-Hecbench-Julia-paper/companion/14_v100_counter_assessment.tex',S/'retained_roi_aggregates/audit.json',S/'benchmark_assessment_snapshot/descriptor_assessment.json',R/'ontology/tools/generate_ontology_report.py']
for b in ['triad','babelstream','jacobi','zerocopy']:
 files.extend([R/f'ontology/studies/julia_abstractions_537/domains/{b}.json',S/f'benchmarks/{b}.json'])
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
a=json.loads(files[2].read_text());d=json.loads(files[3].read_text());rows=[]
for r in a['records']:
 t=r['totals'];rows.append({k:r[k] for k in ['benchmark','model','configuration_id','selected_profile','expected_scientific_launches','accepted_complete_scientific_kernel_family','complete_elapsed_host_ROI','causal_bottleneck']}|{'totals':t,'source_launch_proof':r['source_launch_proof'],'capture_provenance':r['capture_provenance']})
# Frozen descriptive counts only, never an arithmetic intensity average or causal inference.
groups=[]
for fam in sorted({r['scientific_family'] for r in d['records']}):
 for model in d['denominators']['models']:
  rs=[r for r in d['records'] if r['scientific_family']==fam and r['model']==model]
  groups.append({'scientific_family':fam,'model':model,'denominator_names':len(rs),'assessment_category_counts':dict(collections.Counter(r['assessment_category'] for r in rs))})
e={'schema_version':1,'review_scope':'existing evidence only; no fresh counter execution or causal inference','snapshot_created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_sha256':hashes,'strict_family_records':rows,'frozen_family_coverage':groups,'denominators':d['denominators'],'snapshot_per_model':d['per_model'],'complete_elapsed_ROI_count':0,'causal_assignments':0,'source_inferred_resource_bias_is_not_counter_evidence':True}
(O/'evidence_snapshot.json').write_text(json.dumps(e,indent=2)+'\n')
with (O/'strict_family_coordinates.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['benchmark','model','launches','counted_FP32','DRAM_bytes','counted_FP32_per_DRAM_byte','cold_replay','whole_elapsed_ROI','causal_bottleneck'])
 for r in rows:w.writerow([r['benchmark'],r['model'],r['expected_scientific_launches'],r['totals']['fp32_executed_flops'],r['totals']['dram_bytes'],r['totals']['fp32_flop_per_DRAM_byte'],True,False,'unknown'])
# Fail closed if inputs changed during extraction; parent campaign may update selected metadata.
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items()), 'inputs changed during review extraction'
print(json.dumps({'inputs':len(hashes),'strict_records':len(rows),'family_model_cells':len(groups),'evidence_sha256':hashlib.sha256((O/'evidence_snapshot.json').read_bytes()).hexdigest()}))
