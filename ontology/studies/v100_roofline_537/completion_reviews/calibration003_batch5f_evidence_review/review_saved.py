from pathlib import Path
import json,hashlib,gzip,collections,datetime
D=Path(__file__).resolve().parent;P=Path('/tmp/roofline-completion-20261021/roi_membership_source_batch5_f');S=Path('/home/imo/HeCBench_julia');C=Path('/tmp/hecbench-roofline-v100-20261020/method_calibration_ncu2025_2_0_003/profile_provenance.json')
def sha(b):return hashlib.sha256(b).hexdigest()
raw=(P/'proposed_stage_patches.json').read_bytes();v=json.loads(raw);cache={};issues=[];lines_checked=0;bindingchecks=0
# Independent read-only evidence consistency checks, not candidate execution.
def evidence(z):
 global lines_checked,bindingchecks
 p=Path(z['path']);h=z.get('sha256')or z.get('source_sha256')
 if str(p)not in cache:cache[str(p)]=p.read_bytes()
 b=cache[str(p)];lines=b.decode().splitlines()
 if h and sha(b)!=h:issues.append({'path':str(p),'issue':'source SHA mismatch'});return
 bindingchecks+=1
 checks=[]
 if 'line'in z and'text'in z:checks.append((z['line'],z['text']))
 for label in ['timer_lines','arithmetic_branch_atomic_lines','line_evidence']:checks+= [(x['line'],x['text'])for x in z.get(label,[])]
 for n,t in checks:
  lines_checked+=1
  if not 1<=n<=len(lines)or lines[n-1]!=t:issues.append({'path':str(p),'line':n,'issue':'quoted line mismatch'})
 if 'context'in z:
  n=z['context_start_line'];lines_checked+=1
  if not '\n'.join(lines[n-1:]).startswith(z['context']):issues.append({'path':str(p),'issue':'context mismatch'})
 if 'snippet'in z:
  lines_checked+=1
  if b[z['start_byte']:z['end_byte']].decode()!=z['snippet']:issues.append({'path':str(p),'issue':'snippet mismatch'})
counts=collections.Counter();claims=[]
for patch in v['patches']:
 key=patch['record_key'];counts['models']+=1
 for op in patch['stage_patches']:
  u=op['updates'];pr=u.get('ROI_membership_proof',{});counts['stage_updates']+=1;counts['inclusions'if u.get('semantic_family_in_primary_ROI')is True else'exclusions']+=1
  if pr.get('sample_in_primary_timed_invocation')not in [None,False]or pr.get('whole_elapsed_ROI_claim',False)or pr.get('complete_ROI',False)or pr.get('causal_bottleneck','unknown')not in ['unknown',None]:issues.append({'stage':op['stage_id'],'issue':'unwarranted phase/ROI/causal promotion'})
  src=pr.get('exact_stage_callsite')
  if src:evidence(src)
  for z in pr.get('timed_closure_boundaries',[])+pr.get('evidence',[]):evidence(z)
  claims.append({'benchmark':key['benchmark'],'model':key['model'],'stage_id':op['stage_id'],'included':u.get('semantic_family_in_primary_ROI'),'reason':pr.get('reason')})
notes=json.loads((P/'companion_scientific_observations.json').read_bytes())
for n in notes['observations']:
 for z in n['source_bindings']+[n['scientific_intent_binding']]:
  p=S/z['path'];b=p.read_bytes();cache[str(p)]=b
  if sha(b)!=z['sha256']:issues.append({'path':str(p),'issue':'companion observation binding SHA mismatch'})
for p,b in cache.items():
 if Path(p).read_bytes()!=b:issues.append({'path':p,'issue':'changed during review'})
if (P/'proposed_stage_patches.json').read_bytes()!=raw:issues.append({'issue':'packet changed during review'})
cal=C.read_bytes();cp=json.loads(cal)
(D/'calibration003_snapshot.json').write_bytes(cal)
report={'status':'PRIVATE_CPU_SCIENTIFIC_AND_QUOTATION_REVIEW_NOT_MERGE_ENGINE_VALIDATION','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'packet':{'path':str(P/'proposed_stage_patches.json'),'sha256':sha(raw)},'counts':dict(counts),'names':sorted({p['record_key']['benchmark']for p in v['patches']}),'source_quote_checks':{'unique_files':len(cache),'binding_visits':bindingchecks,'exact_line_context_snippet_checks':lines_checked,'issues':issues},'stable_source_bindings':{p:sha(b)for p,b in cache.items()},'calibration003':{'path':str(C),'snapshot_sha256':sha(cal),'actual_status':cp['status'],'cases':cp['cases'],'all9_success':False,'admission_evidence_prepared':False,'reason':'Actual status prepared/cases empty at snapshot; no passed case evidence exists yet.'},'scientific_findings':[{'case':'f16atomic/f16max','assessment':'Timing callbacks contain actual packed atomic and packed/scalar max kernels; warmup/reset/readback excluded as stated. Source packed arithmetic is not a tensor or instruction-count claim.'},{'case':'f16sp','assessment':'Three custom modes across five grids plus vendor calls within execute are timed. Shared family aliases must not assign sampled accumulator type or vendor symbol; retained placeholders remain unknown.'},{'case':'f8cast','assessment':'Four encoding branches timed, guard resets before timer. Named encoding policies and original precision bits preserved; no FP8 hardware GEMM inference.'},{'case':'fft','assessment':'Forward+inverse pair callback in timer; extra forward spectrum/Parseval call after timer. Same family may appear outside timed phase; alias interpretation appropriately source-only.'},{'case':'face','assessment':'Timed detect pipeline includes allocation/address materialization/host classifier/grouping; separate replay captures detail. Source family proves only address-kernel membership, not full GPU classifier.'},{'case':'fhd','assessment':'Factory preparation outside trial timer; chunk constant upload+actual kernel callback inside timer. LLVM constant symbol/Float32 fast-trig description supported, no causal bottleneck claim.'},{'case':'filter/flip','assessment':'Filter counter reset is timed; destination initialization/readback outside. Flip warmup callsites outside timer; repeated callback inside. Kernel signature membership cannot assign actual sample phase.'},{'case':'fdtd3d','assessment':'Withheld because no canonical parent/stages. Do not infer failure or invent membership from saved symbols.'}],'root_merge_validation_still_required':['Current canonical parent/stage hashes and selected capture joins under metadata lock','Full manifest/member/regex uniqueness and actual saved family identity validation by root engine','Recheck selected source stability at admission; this report does not update shared authority'],'gpu_runs':0,'builds':0,'tests':0,'shared_mutations':0}
(D/'review.json').write_text(json.dumps(report,indent=2)+'\n');(D/'stage_claims.json').write_text(json.dumps(claims,indent=2)+'\n');print(json.dumps({'counts':dict(counts),'quote_issues':len(issues),'files':len(cache),'quotes':lines_checked,'calibration_status':cp['status'],'cases':list(cp['cases'])}))
