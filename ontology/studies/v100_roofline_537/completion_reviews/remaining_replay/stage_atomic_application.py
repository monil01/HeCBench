#!/usr/bin/env python3
"""CPU-only 4model atomicSystemWide full-application replay plan+methodreview.
No GPU, target, compiler or tests are executed. Current kernelhold untouched.
"""
import pathlib,json,hashlib,copy,argparse,sys
P=pathlib.Path;W=P('/home/imo/HeCBench_julia');BASE=P('/tmp/roofline-method-review/reviewed_plan_v4_plus_v5_cap512/atomicSystemWide');HERE=P(__file__).resolve().parent
GUIDE=P('/usr/share/doc/nsight-compute/ProfilingGuide/index.html')
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def dump(p,a):P(p).write_text(json.dumps(a,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-root',type=P,required=True);ap.add_argument('--review-output',type=P,required=True);a=ap.parse_args();out=a.output_root.resolve();reviewpath=a.review_output.resolve()
 if out.exists()or out==P('/tmp')or not out.is_relative_to('/tmp')or reviewpath.exists()or not reviewpath.is_relative_to('/tmp')or reviewpath.is_relative_to(out):ap.error('Fresh /tmp planroot and separate fresh review file required')
 source=HERE/'atomic_application_pass.py';out.mkdir();models={};plans=[];guidebytes=GUIDE.read_bytes();start=guidebytes.find(b'In addition, application replay can support');end=guidebytes.find(b'</p>',start)
 if start<0 or end<0:raise ValueError('Exact installedguide supportparagraph required')
 guideproof={'path':str(GUIDE),'sha256':sha(GUIDE),'byte_offset_start':start,'byte_offset_end_exclusive':end,'exact_HTML_utf8':guidebytes[start:end].decode(),'section':'2.3.2 Application Replay','source_version_qualification':'InstalledNCU2022.4.1 guide; no externalproductversion assumption.'}
 for m in ['cuda','cuda_jl','ka','jacc']:
  bp=BASE/m/'plan.json';b=json.loads(bp.read_text());slot=out/'atomicSystemWide'/m;scratch=slot/'scratch';slot.mkdir(parents=True);scratch.mkdir();(scratch/'profile_outputs').mkdir();(scratch/'ncu_tmp').mkdir();(scratch/'profile_source').mkdir();plan=copy.deepcopy(b)
  if b['benchmark']!='atomicSystemWide'or b['blocking_gates']or b['runtime_asset_bytes']:raise ValueError('Exact ready zero-asset atomic plan required')
  bindings=dict(b['execution_bindings'])
  for p,h in bindings.items():
   if sha(p)!=h:raise ValueError('Canonical source drift '+p)
  rp=P(b['reference_result']);cp=rp.with_name('config.json')
  if sha(rp)!=b['reference_result_sha256']or sha(cp)!=b['raw_config_sha256']:raise ValueError('Reference/config drift')
  old=b['scratch_cwd']
  def move(v):
   if isinstance(v,str):return v.replace(old,str(scratch))
   if isinstance(v,list):return [move(z)for z in v]
   if isinstance(v,dict):return {k:move(z)for k,z in v.items()}
   return v
  for k in ['scratch_cwd','environment_overrides','build_command','profile_target_command']:plan[k]=move(b[k])
  wrapper=scratch/'profile_source'/'atomic_application_pass.py';wrapper.write_bytes(source.read_bytes())
  # Resolve the same installed Julia version directly, removing launcher children
  # as an application-replay matching covariate; sourceprojects/arguments unchanged.
  real=P('/home/imo/.julia/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia')
  child=[str(real)if x in['julia','/home/imo/.juliaup/bin/julia']else x for x in plan['profile_target_command']]
  if m!='cuda':bindings[str(real)]=sha(real)
  plan['profile_target_command']=[sys.executable,str(wrapper),*child]
  for p in [bp,rp,cp,source,P(__file__).resolve(),GUIDE,W/'tools/v100_profile_write_guard.py',wrapper]:bindings[str(p)]=sha(p)
  for directory in [W/'src/atomicSystemWide-julia',W/'src/atomicSystemWide-cuda']:
   for p in directory.rglob('*'):
    if p.is_file()and p.suffix in ['.jl','.cu','.cpp','.h','.cuh','.py','.hpp']:bindings[str(p)]=sha(p)
  for name in ['study_native.cu','study_common.jl']:
   p=W/'src/atomicSystemWide-julia'/name;text=p.read_text()
   if name=='study_native.cu'and 'for(int tr=1;tr<=5;tr++)'not in text:raise ValueError('Native5trials sourceproof drift')
   if name=='study_common.jl'and ('for trial in 1:5'not in text or'reset!();complete!()'not in text):raise ValueError('Julia5trials+warmup sourceproof drift')
  key='atomicSystemWide/'+m
  record={'benchmark':'atomicSystemWide','model':m,'configuration_id':b['configuration_id'],'selected_configuration_id':b['configuration_id'],'source_result_sha256':b['reference_result_sha256'],'reference_result_sha256':b['reference_result_sha256'],'source_config_sha256':b['raw_config_sha256'],'raw_config_sha256':b['raw_config_sha256'],'selected_result':str(rp),'selected_config':str(cp),'execution_bindings':bindings,'replay_mode':'application','app_replay_match':'grid','source_reviewed':True,'runtime_validated':False,'guide_proof':guideproof,'rationale':'Currentkernelreplaycannotre-execute concurrentCPUatomic participant. Installedguide supports fullapplication replayhostinterdependencies. Freshprocessreruns actualGPUkernel plusoriginalCPUatomics,resetandall5checks eachpass; no toolsave/restore.','launch_proof':{'scientific_launches_per_full_process':6,'source_sequence':['untimedreset+launch+CPU8participants+devicefence','trial1reset+launch+CPU8participants+devicefence+fullpropertycheck','trial2same','trial3same','trial4same','trial5same'],'geometry':{'grid':[1,1,1],'threads':[8,1,1],'GPU_participants':8,'CPU_participants':8,'loops_per_participant':1000},'context_stream':'OnevisibleGPU6,oneexplicitcontext/currentdefaultorframeworkstream; no userdynamicstreamdispatch.','matching_qualification':'Scientificfamilyfixed6calls; additionalframeworksetup kernels notcountedwithoutactualcensus. Actualmanglednames/passmatching/selectorcounts stillrequirecapture.'},'correctness_proof':{'original_trials':5,'checked_values_per_trial':42,'artifact_words_per_trial':55,'artifact_bytes_per_trial':220,'accepted_contract':'Propertyscores+boundedExch/CAS+32fullguards, notnumericalrawstateidentity.','per_pass_wrapper':'Runsunchangedwholechild; retains fullperpassstdout,rawchildstatus/signal,all5privateartifacts,independentlychecksrawpropertyvectorandcompleteartifactbytes.','output_isolation':'STUDY_OUTPUT_DIR/application_passes/pass_PID_UUID; uniqueperapprerun, nooverwrite. WrapperdoesnotchooseGPUinputs/launches/CPUordering.'},'limitations':['source-reviewed methodcandidate,actualtargetruntimeandNCUmatchedpasses unvalidated','Countersinteger/systematomics; nofabricatedFPwork orHBM-onlyroof','Originalhost/GPUoverlapandmanagedpageresidency notquantitativelypreservedby sourceproof','CurrentgenericatomicSystemWidekernelhold remains unlessexplicitreviewedapplicationoptinpassesallbindings','Applicationreplay duplicateswhole5trialhost/auditcost; timeoutsneedfresh attempt, notchecksuppression','Priorblockedkernelreplayattempts remainimmutable']}
  models[key]=record;plan['execution_bindings']=dict(bindings);plan['staged']=True;plan['application_replay_review']={'path':str(reviewpath),'sha256':None,'entry':key};plan['application_replay_context_proposal']={'replay_mode':'application','app_replay_match':'grid','cache_control':'all explicitcollectorpolicy, notmanagedresidencypreservation','runtime_validated':False,'primary_contract_unchanged':True};plan['commands_executed']=[];plans.append((slot,plan))
 review={'schema_version':1,'models':models,'installed_guide':guideproof,'scope':'AtomicSystemWideonly exactselected fullapplication replaycandidate. Sourcequalificationnotmeasurementcompletion; generickernelholdunchanged.','source_reviewed':True,'runtime_validated':False,'GPU_runs':0,'build_runs':0,'tests':0};dump(reviewpath,review);h=sha(reviewpath)
 summary=[]
 for slot,plan in plans:
  plan['application_replay_review']['sha256']=h;dump(slot/'plan.json',plan);summary.append({'model':plan['model'],'plan':str(slot/'plan.json'),'sha256':sha(slot/'plan.json'),'review_sha256':h})
 dump(out/'summary.json',summary);print(json.dumps({'staged':len(plans),'review':str(reviewpath),'review_sha256':h,'GPU_runs':0,'build_runs':0}))
if __name__=='__main__':main()
