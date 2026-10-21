"""Private read-only targeted blocked-slot/queue review; no study imports/GPU."""
from pathlib import Path
import json,hashlib,fcntl,os,datetime,collections,csv,re
R=Path('/home/imo/HeCBench_julia');S=R/'ontology/studies/v100_roofline_537';RAW=Path('/tmp/hecbench-roofline-v100-20261020');O=Path(__file__).parent;inputs={}
def sha(b):return hashlib.sha256(b).hexdigest()
def need(x,s):
 if not x:raise ValueError(s)
def read(p):
 p=Path(p);b=p.read_bytes();inputs[str(p)]=b;return b
def j(p):return json.loads(read(p))
def proc(pid,start):
 p=Path('/proc')/str(pid)
 try:
  v=(p/'stat').read_text().rsplit(')',1)[1].split();args=[x.decode()for x in(p/'cmdline').read_bytes().split(b'\0')if x];state={'pid':pid,'start_ticks':v[19],'state':v[0],'uid':p.stat().st_uid,'argv':args,'children':(p/'task'/str(pid)/'children').read_text().split()};state['exact_generation_live_owned']=v[19]==start and v[0]not in ['Z','X']and p.stat().st_uid==os.getuid();return state
 except(FileNotFoundError,ProcessLookupError):return{'pid':pid,'expected_start_ticks':start,'exact_generation_live_owned':False,'state':'absent'}
def values(argv,flag):return[argv[i+1]for i,t in enumerate(argv[:-1])if t==flag]
D=S/'completion_reviews/completion_supervisor';route_specs={'023':(1157236,'76188511',RAW/'batch023-completion-a'),'028':(1182549,'76316068',RAW/'batch028-remaining-paths'),'050':(1326488,'77008700',RAW/'batch050-rbc-retained-samples'),'054':(1367433,'77213195',RAW/'batch054-tsne-checkers-complete')}
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
with Path('/tmp/hecbench-roofline-metadata.lock').open('r')as lock:
 fcntl.flock(lock,fcntl.LOCK_SH)
 initial=j(O/'initial_snapshot.json');progress=j(S/'PROGRESS.json');config=j(RAW/'completion_20261021_v15/configuration.json');adoption=j(D/'root_adoption_command.json');need(config['output_root']==str(RAW/'completion_20261021_v15'),'Rootfrontier drift');expected_ids={(x['pid'],str(x['start_ticks']))for x in config['controllers']};need(len(expected_ids)==40,'Unexpected controller count')
 retry_source=read(R/'tools/v100_roofline_retry.py').decode();match=re.search(r"STANDARD=set\('([^']+)'\.split\(\)\)",retry_source);need(match,'Retry whitelist not recognized');standard=set(match.group(1).split())
 routes={}
 for name,(pid,start,folder)in route_specs.items():
  need((pid,start)in expected_ids,'Route not in monitored root40');q=proc(pid,start);need(q['exact_generation_live_owned'],'Route generation no longer live '+name);need(str(folder)in values(q['argv'],'--plan-root'),'Route does not traverse exact batch');need(any(Path(a).name=='v100_roofline_collect.py'for a in q['argv']),'Not collector');routes[name]=q
 rows=[];categories=collections.Counter();source_checks=[]
 two=j(S/'completion_reviews/two_device_applicability_review/review.json');twomap={(r['benchmark'],r['model']):r for r in two['records']};ar=j(S/'completion_reviews/allreduce_two_device/root_cpu_review.json');need(ar['no_GPU5_permission_inferred']is True,'Unexpected permission premise');need(sha(read(ar['proposal']['path']))==ar['proposal']['sha256'],'Allreduce proposal binding drift')
 # Recheck existing two-device source authority, without inferring execution/counter absence.
 for review in two['records']:
  for path,decl in review['source_bindings'].items():
   if decl.get('expected'):
    need(sha(read(path))==decl['expected'],'Two-device source binding drift '+path)
  for e in review['source_line_evidence']:
   source=e['source'];raw=read(source['path']);need(sha(raw)==source['sha256'],'Quotedsource hash');lines=raw.decode().splitlines();start=e['first_line'];need('\n'.join(lines[start-1:]).startswith(e['snippet']),'Quoted source bytes mismatch '+source['path'])
   source_checks.append({'benchmark':review['benchmark'],'model':review['model'],'path':source['path'],'sha256':source['sha256'],'first_line':start,'scope':'exact supplied source quote, possibly capped longline; not GPU absence proof'})
 snapshotrows=[]
 for p in sorted((S/'benchmarks').glob('*.json')):
  raw=read(p);ledger=json.loads(raw)
  for m,state in ledger['models'].items():
   if state['profiling_status']=='not_started_plan_blocked':snapshotrows.append((ledger['benchmark'],m,state,ledger))
 need(len(snapshotrows)==43,'Current count changed; fresh scope review required')
 for b,m,state,ledger in snapshotrows:
  row={'benchmark':b,'model':m,'selected_status':state['profiling_status'],'canonical_gates':state.get('profile_plan_blocking_gates',[]),'configuration_id':ledger['accepted_named_contract']['configuration_id'],'current_finite039_041_eligible':False,'finite_retry_reason':'Current selected status is not profile_process_failed with frozen externalGPU/walltimeout parent; no eligibility is inferred from being blocked.'}
  if b in ('allreduce','ccl','p2p','pingpong','ring'):
   if b=='allreduce':
    category='two_device_explicit_permission_and_private_method_hold';row.update(existing_authority_path=str(S/'completion_reviews/allreduce_two_device/root_cpu_review.json'),existing_authority_status=ar['status'],action='Retain pending explicitGPU5authorization; reviewed immutable-reference private rank1-add method is separate from full collectiveROI. No new job or alternative arithmetic proposed.')
   elif b=='ccl':
    rev=twomap[(b,m)];category='two_device_vendor_reduction_counter_method_hold';row.update(existing_authority_path=str(S/'completion_reviews/two_device_applicability_review/review.json'),source_applicability=rev['scalar_roofline_applicability'],action='Needs genuinely reviewed two-device vendorFP32reduction profiling method and explicitGPU5authorization. Ordinary one-device retry or allreduce private-add scope cannot substitute.')
   else:
    rev=twomap[(b,m)];category='two_device_source_transport_scalar_NA_separate_tier';row.update(existing_authority_path=str(S/'completion_reviews/two_device_applicability_review/review.json'),source_applicability=rev['scalar_roofline_applicability'],action='Existing source primarytimed transport scalar-roofline NA is already annotated. Retain process planhold; no no-device-kernel/no-SM-activity or zero-metrics claim. Communication/remote-access tier requires separate method/permission.')
   row.update(category=category,live_route=None,uncovered_fixable_omission=False);categories[category]+=1;rows.append(row);continue
  if b=='gc'or b=='memcpy':name='023'
  elif b=='rayleighBenardConvection':name='050'
  elif b=='tsne':name='054'
  else:name='028'
  pid,start,folder=route_specs[name];path=folder/b/m/'plan.json';raw=read(path);plan=json.loads(raw);need(plan['configuration_id']==row['configuration_id'],'Queued contractCID differs');need(plan['staged']is True and not plan['blocking_gates'],'Queued plan still gated');need(plan['raw_config_sha256']==ledger['accepted_named_contract']['raw_config_sha256']and plan['reference_result_sha256']==ledger['accepted_named_contract']['result_sha256'],'Queued selected config/result mismatch');need(b not in values(routes[name]['argv'],'--exclude-benchmark'),'Queued collector excludes benchmark')
  bm=values(routes[name]['argv'],'--benchmark');need(not bm or b in bm,'Queued collector filteredout benchmark')
  bindings_reviewed=[]
  # Source-only inspection: large binary fixtures are left to normal runtime full-binding checks.
  for source,digest in plan['execution_bindings'].items():
   q=Path(source)
   if q.is_file()and q.suffix in ('.py','.jl','.cu','.cuh','.h','.cpp','.toml')and q.stat().st_size<=4*1024*1024:
    need(sha(read(q))==digest,'Queued source drift '+source);bindings_reviewed.append({'path':source,'sha256':digest})
  gate=plan.get('profile_window_review')
  if gate:
   need('--profile-window-reviewed'in routes[name]['argv'],'Missing explicit marker-window optin');need(sha(read(gate['path']))==gate['sha256'],'WindowreviewSHA')
  rawprovpath=path.parent/'profile_provenance.json'
  rawstate=json.loads(rawprovpath.read_bytes()).get('status')if rawprovpath.is_file()else'plan_only_unstarted'
  row.update(category='explicit_live_queued_repair_not_success_claim',live_route={'identity':str(pid)+':'+start,'plan_path':str(path),'plan_sha256':sha(raw),'batch':folder.name,'raw_status_snapshot':rawstate,'matching_root_frontier':True,'source_bindings_inspected':bindings_reviewed},queued_plan_blocking_gates=plan['blocking_gates'],queued_extra_fields_requiring_generic_retry_review=sorted(set(plan)-standard),uncovered_fixable_omission=False,action='Allow already queued source-bound attempt; do not launch duplicate. A future failure requires actualoutcome/method-preservation review, not assumed039/041recovery.')
  categories[row['category']]+=1;rows.append(row)
 # Read reviewed055 follower as an additional finite route for RBC, not duplicate independent launch.
 f=adoption['rbc055_follower'];need((f['pid'],str(f['start_ticks']))in expected_ids,'055notrootfrontier');fq=proc(f['pid'],str(f['start_ticks']));need(fq['exact_generation_live_owned'],'055generation notlive');lrpath=Path('/tmp/roofline-completion-20261021/rbc055_guard_recovery/launch_review.json');lrraw=read(lrpath);need(sha(lrraw)==values(fq['argv'],'--review-sha256')[0],'055reviewSHA');lr=json.loads(lrraw);need(lr['predecessor']['pid']==1326488 and lr['predecessor']['start_ticks']=='77008700','055predecessor mismatch');follower_source=read(fq['argv'][1]);need(sha(follower_source)==lr['follower_sha256'],'055follower source changed');need(lr['collector_argv'][lr['collector_argv'].index('--plan-root')+1]==str(RAW/'batch055-rbc-retained-guard-retry'),'055exactbatch')
 for row in rows:
  if row['benchmark']=='rayleighBenardConvection':row['additional_reviewed_finite_route']={'identity':str(f['pid'])+':'+str(f['start_ticks']),'review_path':str(lrpath),'review_sha256':sha(lrraw),'scope':'One-shot follows exact050generation; runtime success unproved, no duplicate launch proposed.'}
 # GCJulia fixture uses canonical READ path; native052 relocation repair is not missing from those entries.
 gcpath=RAW/'batch023-completion-a/gc/ka/scratch/profile_source/study_common.jl';gcraw=read(gcpath);gct=gcraw.decode();need('read(joinpath("/home/imo/HeCBench_julia/src/gc-julia","fixture_csr.bin"))'in gct,'GCJulia fixture role changed');need((R/'src/gc-julia/fixture_csr.bin').is_file(),'GCJulia canonical fixture absent')
 gcproof={'path':str(gcpath),'sha256':sha(gcraw),'readonly_fixture_role':'canonical src/gc-julia/fixture_csr.bin read, not relocated __FILE__ directory lookup','fresh_sidecar_role':'profile_outputs/gc_sidecars under this exact scratch, existing guard applies','new_native052_fix_needed_for_Julia':False}
 # Source proof of reviewed custom builders: no new output rewrite required.
 builderproof=[]
 for b in ['remap','sc']:
  q=R/'src'/(b+'-julia')/'study_build.py';source=read(q).decode();need("'-o','study_native'"in source or"'-o','libsc_cpu.so'"in source,'Unexpected builder output pattern');builderproof.append({'benchmark':b,'path':str(q),'sha256':sha(source.encode()),'observed_source_policy':'Canonical __file__ used only source path; build -o targets relative new scratch. No builder execution in review.'})
 stable_paths=list(inputs)
 for path in stable_paths:need(Path(path).read_bytes()==inputs[path],'Input changed during SHsnapshot '+path)
 finished=datetime.datetime.now(datetime.timezone.utc).isoformat();final={'progress_updated_at':progress['updated_at'],'selected_count':len(rows),'supervisor_output_root':config['output_root'],'same_monitored40':len(expected_ids),'selected_status_counts':progress['status_counts']};need(progress['status_counts']['not_started_plan_blocked']==len(rows),'PROGRESS/ledger count mismatch')
report={'status':'all43_current_planblocked_assessed_no_new_uncovered_single_device_omission','snapshot_started_at':started,'snapshot_finished_at':finished,'metadata_method':'ExistingLOCK_SH over canonicalstate/sourcebindings/controllerreview read; beforeafterSHA stable. /proc exact generations are point-in-time observations, not future liveness promise. Rawactive status snapshots may progress.','snapshot':final,'category_counts':dict(categories),'uncovered_repairable_omission_count':0,'new_jobs_or_patches_proposed':0,'current_generic039_041_candidates_among43':0,'rows':rows,'primary_live_routes':routes,'GC_Julia_fixture_review':gcproof,'reviewed_builder_policy':builderproof,'two_device_source_quote_checks':source_checks,'qualifications':['Queued23 is not23successfulprofiles. Future failures or changed source/selection require new review.','12transport source scalar-NA records do not prove noGPUkernels, noSMactivity, rawcounterzero, or wholeprogramabsence.','ccl4needs realvendorFP32reduction method; allreduce4has separately scopedprivate method butGPU5permissionpending. No authorization inferred.','Initial43 planblocked labels retain historicalplanassessment while repairedplans are queued. Do not clearstatus/emitfakemetrics just from this review.','Finite039/041 selects only frozen selected operational stops and conservatively holds custommethods/signatures. It must not be counted as coverage of every currentlyblocked/futurefailedslot.','Custom056–058 are reviewedrootqueue recovery for otherselectedfailures, not extra solutions needing duplicate launches for these43.','No extra helper/patch justified: all potentialsingle-deviceomissions inspected already have source-bound queued plans.'],'input_sha256':{p:sha(b)for p,b in inputs.items()},'newGPU_build_tests_imported_studycode_shared_mutations':0}
(O/'review.json').write_text(json.dumps(report,indent=2)+'\n')
with(O/'slots.csv').open('w',newline='')as file:
 fields=['benchmark','model','selected_status','category','configuration_id','canonical_gates','live_route','additional_reviewed_finite_route','queued_extra_fields_requiring_generic_retry_review','action'];writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore');writer.writeheader()
 for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True)if isinstance(v,(list,dict))else v for k,v in row.items()if k in fields})
print(json.dumps({'review_sha256':sha((O/'review.json').read_bytes()),'category_counts':dict(categories),'uncovered_repairable':0,'selected_count':len(rows),'finished_at':finished}))
