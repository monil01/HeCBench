#!/usr/bin/env python3
"""Read frozen/completed profile evidence only; writes its own audit directory."""
import collections,csv,datetime,hashlib,io,json,pathlib,subprocess
R=pathlib.Path('/tmp/hecbench-roofline-v100-20261020')
S=pathlib.Path('/home/imo/HeCBench_julia')
D=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):
 p=pathlib.Path(p);return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def dump(n,x):(D/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
rows=[];bad=[];nulls=collections.Counter();warningcounts=collections.Counter();metricnulls=collections.Counter()
for p in sorted(R.glob('batch*/*/*/profile_provenance.json')):
 raw=p.read_bytes();x=json.loads(raw);pr=x.get('profile',{});row={'profile':str(p),'status':x.get('status'),'kernel_count':x.get('kernel_count'),'started_at':x.get('started_at'),'finished_at':x.get('finished_at'),'execution_started_at':pr.get('started_at'),'wall_seconds':pr.get('wall_seconds'),'stop_reason':pr.get('stop_reason'),'returncode':pr.get('returncode'),'error':x.get('error'),'counter_parse_error':x.get('counter_parse_error')}
 row['snapshot_sha256']=hashlib.sha256(raw).hexdigest();row['snapshot_is_finished']=bool(x.get('finished_at'));row['plan_binding']=bind(p.parent/'plan.json')
 if x.get('finished_at'):
  row['closed_evidence']=[bind(q)for q in [p,p.parent/'counters.csv',p.parent/'execution.log',p.parent/'derived.json']if q.exists()]
  if x.get('status')=='profile_process_completed' and not x.get('kernel_count'):bad.append(row)
  q=p.parent/'derived.json'
  if x.get('status')=='profile_process_completed' and q.exists():
   d=json.loads(q.read_text());row['derived_kernel_count']=len(d['records'])
   for z in d['records']:
    warningcounts.update(z.get('warnings',[]))
    for k in ['duration_seconds','dram_bytes','observed_integer_predicated_on_thread_instructions']:
     if z.get(k) is None:nulls[k]+=1
   for g in d.get('raw_metrics',[]):
    for k,v in g['metrics'].items():
     if v is None:metricnulls[k]+=1
 rows.append(row)
dump('attempt_snapshot.json',{'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'All batch*/*/* profile provenance present at scan; in-flight files are snapshots, not immutable receipts. No GPU/build or source/ledger mutations.','counts':dict(collections.Counter(x['status']for x in rows)),'rows':rows})
dump('parser_and_zero_kernel_review.json',{'completed_zero_kernel_cases':bad,'completed_required_null_counts':dict(nulls),'parsed_null_metric_columns':dict(metricnulls),'warnings':dict(warningcounts),'interpretation':'Text device/cache/NVLink columns are coerced to null in numeric metric dictionary but preserved in raw CSV; not malformed numerical counters. Integer counter missing in older cohorts is unknown, never zero.','proposals':[{'predicate':'returncode==0 and stop_reason is None and parsed_kernel_count==0','status':'profile_completed_no_profiled_kernels','note':'Explicit no-device-kernel sources may be not-applicable; other cases require filter/child/injection review. Never populate zero FLOPs or claim profiling completeness.'},{'predicate':'number(raw_value) is None on an explicitly requested numeric field','action':'Retain raw token, metric name/unit/kernel identity and conversion reason alongside null; never substitute zero or silently omit the kernel.'}]})
helptext=subprocess.run(['/usr/bin/ncu','--help'],capture_output=True,text=True,check=True).stdout
(D/'ncu_help.txt').write_text(helptext)
(D/'ncu_version.txt').write_text(subprocess.run(['/usr/bin/ncu','--version'],capture_output=True,text=True,check=True).stdout)
sourcefiles=['tools/v100_roofline_collect.py','tools/v100_roofline_classify.py','tools/v100_roofline_followups.py','tools/v100_roofline_summarize.py','src/slu-julia/integrated_numeric.cuh','src/slu-julia/study_common.jl','src/slu-julia/slu-julia-cuda.jl/main.jl','src/slu-julia/slu-julia-KA/main.jl','src/slu-julia/slu-julia-JACC/main.jl','src/assert-julia/run_study.py']
dump('source_bindings.json',{'files':[bind(S/p)for p in sourcefiles],'selector_proof':[bind(R/'selector_probe'/n)for n in ['selector_shapes.cu','shape_kernelid_provenance.json','shape_kernelid.csv','shape_combined_provenance.json','shape_combined.csv']],'help':bind(D/'ncu_help.txt'),'version':bind(D/'ncu_version.txt')})
print(json.dumps({'profiles':len(rows),'statuses':dict(collections.Counter(x['status']for x in rows)),'zero_kernel_success':len(bad),'required_nulls':dict(nulls)}))
