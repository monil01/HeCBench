from pathlib import Path
import json,gzip,hashlib,collections
D=Path(__file__).resolve().parent;R=Path('/home/imo/HeCBench_julia');RAW=Path('/tmp/hecbench-roofline-v100-20261020');rows=json.loads((D/'frozen_failure_rows.json').read_bytes())['rows'];controllers=json.loads((D/'controller_generations.json').read_bytes())['controllers'];routes=collections.defaultdict(list)
def sha(b):return hashlib.sha256(b).hexdigest()
for c in controllers:
 if not c.get('same_generation')or c.get('state')=='Z':continue
 argv=c['argv'];roots=[Path(argv[i+1])for i,a in enumerate(argv[:-1])if a=='--plan-root']
 # Exact reviewed custom future jobs, not guessed from empty raw roots.
 if c['pid']==1451256:
  rev=json.loads(Path('/tmp/roofline-completion-20261021/uncovered_custom_followups_056_058/launch_review.json').read_bytes());roots += [Path(x['plan_root'])for x in rev['cases']]
 if c['pid']==1575364:roots += [RAW/'batch059-miniDGS-ncu2025',RAW/'batch060-atomic-app-ncu2025']
 for root in roots:
  for p in root.glob('*/*/plan.json'):
   plan=json.loads(p.read_bytes());key=(plan['benchmark'],plan['model']);routes[key].append({'controller_identity':str(c['pid'])+':'+c['start_ticks'],'plan':str(p),'plan_sha256':sha(p.read_bytes()),'plan_root':str(root),'scope':'Named in live ownedcontrolleror exactreviewedfuturejob; currentindividualcasecursor not inferred. Collector may skipqualified.','existing_raw_outcome':(p.parent/'profile_provenance.json').exists()})
master=next(x for x in controllers if x['pid']==1213116);argv=master['argv'];excluded={argv[i+1]for i,x in enumerate(argv[:-1])if x=='--exclude-benchmark'};journal=RAW/'campaign_003.jsonl';jb=journal.read_bytes();events=[]
for line in jb.decode().splitlines():
 try:events.append(json.loads(line))
 except ValueError:pass
frozen={x['benchmark']for x in events if x.get('event')=='frozen'};frontier=json.loads((RAW/'retry_021_frontier.json').read_bytes());retry021={(b,x['model'])for b,items in frontier.items()for x in items}
report=[]
for row in rows:
 if row['view']!='selected':continue
 x={k:row[k]for k in ['benchmark','model','category','stop_reason','status','profile_id','generic_retry_status_eligible','custom_plan_fields','custom_methods']};key=(x['benchmark'],x['model']);x['explicit_live_queue_routes']=routes[key];x['existing021_frontier_member']=key in retry021;x['original_campaign_remaining_assessment']=x['benchmark']not in excluded and x['benchmark']not in frozen
 x['finite039041_qualification']='not_status_reason_eligible'
 if row['generic_retry_status_eligible']:
  x['finite039041_qualification']='eligible_reason_but_actual_method_history_deadline_and_freshplan_equivalence_mustpass'
  if row['custom_plan_fields']or row['custom_methods']or x['benchmark']in ['perlin','zmddft','su3','assert','atomicSystemWide','streamUM','rayleighBenardConvection']:x['finite039041_qualification']='source_reviewed_custom_or_nonstandard_selected_method_hold_expected; allhistorymustalsobechecked'
 x['next_action']='Await namedexistingqueue; no duplicate'if x['explicit_live_queue_routes']else 'Await originalremainingassessment or existing021/finite039041 stricteligibility; no automatic compatibility switch'
 if x['benchmark']=='sparkler':x['next_action']='Concrete wrappervisibility mismatch uncovered; root requested separateprivate061 legacy2022 guard-only proposal afterthistriage; no job launched.'
 elif x['benchmark']=='snicit':x['next_action']='Unknown parentrc245/255. Remainingoriginalcampaign assessment is distinct from fresh2025compatibility experiment; preserveallchecks, inspectinnerwrapper/counters/processfailure beforeanymethodchange.'
 report.append(x)
(D/'queue_coverage.json').write_text(json.dumps({'records':report,'campaign_journal':{'path':str(journal),'sha256':sha(jb),'bytes':len(jb)},'campaign_excluded':sorted(excluded),'campaign_frozen_names':len(frozen),'finite_retry_warning':'Reason eligibility is NOT executableeligibility. Currentretry requires selectedfrozenfailed+foreign/timeout, strictallhistoricalmethod/deadlines<=3600 and freshgeneric equivalence; customholdsneverwaived.039041 notexecutedyet; no membership guaranteed.','future_method_preservation_review':'Currentretry METHOD_FIELDS omits profiler_method. Current059/060plans have noncanonical new_profiler_rollout_provenance and are held anyway. For anyfuturestandard-looking2025failedplan, rootshouldexplicitlyreview preservation of profiler_method beforegenericretry; do notassumesamecalibratedmethod fromcanary alone.'},indent=2)+'\n')
print('selected',len(report),'explicitroutes',sum(bool(x['explicit_live_queue_routes'])for x in report),'reasoneligible',sum(x['generic_retry_status_eligible']for x in report),'current_selected_customholds',sum(x['finite039041_qualification'].startswith('source_reviewed')for x in report));print('nonopcoverage',[(x['benchmark'],x['model'],len(x['explicit_live_queue_routes']),x['original_campaign_remaining_assessment'])for x in report if not x['generic_retry_status_eligible']])
