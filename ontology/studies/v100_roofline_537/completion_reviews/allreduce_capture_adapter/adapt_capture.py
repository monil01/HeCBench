#!/usr/bin/env python3
"""PROPOSAL: CPU-only adaptation of an explicitly reviewed allreduce capture.
Not executed or imported by its author. Never launches a process/GPU or writes
source/ontology/ledgers. Root must freeze a prospective review before capture.
"""
import argparse, array, base64, csv, datetime, hashlib, json, math, os, re, struct, sys, types
from pathlib import Path

MODELS=('cuda','cuda_jl','ka','jacc')
UUID='GPU-c122ecbd-d9f3-18a5-42d4-1535af37e56d'
METHOD='allreduce_two_device_rank1_private_add_NCU2022_4_1_v1'
METRICS=['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum','dram__bytes.sum','dram__bytes.sum.peak_sustained','dram__cycles_elapsed.avg.per_second','sm__cycles_elapsed.avg.per_second','lts__t_bytes.sum','sm__inst_executed_pipe_tensor.sum','lts__t_bytes.sum.peak_sustained','lts__cycles_elapsed.avg.per_second','sm__inst_executed_pipe_tensor.sum.peak_sustained','sm__throughput.avg.pct_of_peak_sustained_elapsed','dram__throughput.avg.pct_of_peak_sustained_elapsed','smsp__sass_thread_inst_executed_op_integer_pred_on.sum']+['smsp__sass_thread_inst_executed_op_'+x+'_pred_on.sum'for x in ('hadd','hmul','hfma','fadd','fmul','ffma','dadd','dmul','dfma')]+['sm__sass_thread_inst_executed_op_'+x+'_pred_on.sum.peak_sustained'for x in ('hfma','ffma','dfma')]
PCI=['device__attribute_pci_'+x+'_id' for x in ('domain','bus','device')]
CC=['device__attribute_compute_capability_major','device__attribute_compute_capability_minor']
SIZES=(1,32,33,256,4099,1048576)
COUNT=5269603

def require(ok,why):
    if not ok:raise ValueError(why)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def obj(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def bound(item):
    p=Path(item['path']);require(p.is_file()and sha(p)==item['sha256'],'Missing/changed reviewed file: '+str(p));return p

def archive_tree(root):
    entries=[]
    for p in sorted(root.rglob('*')):
        require(not p.is_symlink(),'Raw evidence symlink not permitted: '+str(p))
        if p.is_file():entries.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':sha(p)})
    return entries

def labelled(log,label):
    result=[]
    for line in log.splitlines():
        if label in line:
            # Accepted MPI logs may prefix a rank label. Require no ambiguous
            # multiple labels or non-whitespace tail after the actual JSON.
            require(line.count(label)==1,'Duplicate marker on line')
            raw=line.split(label,1)[1];value,end=json.JSONDecoder().raw_decode(raw)
            require(not raw[end:].strip(),'Trailing marker bytes')
            result.append(value)
    return result

def exact_artifact(p):
    require(p.stat().st_size==COUNT*4,'Full artifact byte length')
    words=array.array('I');words.frombytes(p.read_bytes())
    if sys.byteorder!='little':words.byteswap()
    def bits(x):return struct.unpack('<I',struct.pack('<f',x))[0]
    pos=0
    def pattern(n,seq,start=0):
        nonlocal pos
        require(all(words[pos+i]==seq[(start+i)%len(seq)]for i in range(n)),'Independent full-value mismatch at section '+str(pos));pos+=n
    def one(x):pattern(1,[bits(x)])
    ranks=[[bits((r+1)*.5+(i-8)/32)for i in range(17)]for r in range(2)]
    sums=[bits(1.5+(i-8)/16)for i in range(17)];sentinel=[bits(-777)]
    one(6)
    for n in SIZES:
        one(n)
        for r in range(2):pattern(n,ranks[r])
    for n in SIZES:
        for r in range(2):pattern(n,sums)
        half=(n+1)//2;tail=n-half
        for r in range(2):
            length=tail if r==0 else half;start=half if r==0 else 0
            pattern(length,ranks[1-r],start);pattern(half-length,sentinel)
        pattern(768,sentinel)
    require(pos==COUNT,'Independent checked count')
    return {'path':str(p),'sha256':sha(p),'bytes':COUNT*4,'checked_values':COUNT,'full_bitwise_oracle_pass':True}

def pcie_tuple(text):
    m=re.fullmatch(r'([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-9a-fA-F]+)',text.strip())
    require(m is not None,'PCI syntax');return tuple(int(x,16)for x in m.groups()[:3])

def validate_slot(slot,plan,prov,bundle,review,parser):
    require(prov['model']==plan['model'],'Parent model join')
    require(prov['status']=='profile_completed_all5checks_and_GPU6identity_verified','Custom capture did not complete')
    require(prov['proposal_sha256']==review['proposal']['sha256'],'Captured proposal identity')
    require(prov['staging_plan_sha256']==sha(slot/'plan.json'),'Actual plan identity')
    require(prov.get('source_bound_before')is True and prov.get('source_bound_after')is True,'Contemporaneous source checks absent')
    require(prov['readonly_bindings_before']==bundle['all_readonly_bindings']==prov['readonly_bindings_after'],'Actual prospective read-only maps before/after')
    for name,key in [('collector_source.py','runner'),('parser_source.py','parser_source'),('calibration_review.json','calibration_review')]:
        require(sha(slot/name)==review[key]['sha256'],'Retained pre-capture source snapshot: '+name)
    require(prov['snapshot_bindings_before']==prov['snapshot_bindings_after'],'Retained snapshot map changed')
    for name,digest in prov['snapshot_bindings_before'].items():
        path=Path(name);require(path.is_relative_to(slot) and path.is_file() and sha(path)==digest,'Actual captured source snapshot binding')
    snapshots=obj(slot/'source_snapshots.json');require(snapshots['created_before_target']is True,'Source snapshots not prospective')
    seen=set()
    for item in snapshots['files']:
        raw=base64.b64decode(item['base64'],validate=True)
        require(item['path']not in seen and hashlib.sha256(raw).hexdigest()==item['sha256'] and len(raw)==item['bytes'],'Lossless source snapshot bytes/SHA')
        require(bundle['all_readonly_bindings'].get(item['path'])==item['sha256'],'Source snapshot not prospectively bound')
        seen.add(item['path'])
    needed={path for path in bundle['all_readonly_bindings']if Path(path).suffix in ('.py','.cu','.h','.cpp','.jl','.toml')}
    require(needed<=seen,'Prospectively bound source omitted from retained snapshots')
    require(prov['primary_timing_replacement']is False,'Profiling cannot replace primary timing')
    profile=prov['profile'];require(profile['returncode']==0 and profile['stop_reason']is None and profile['own_session_quiescent']is True,'Failed or live profiling process')
    require(profile['log_sha256']==sha(slot/'profile.log'),'Actual execution log identity')
    for b in prov['builds']:require(b['returncode']==0 and b['stop_reason']is None and b['own_session_quiescent']is True,'Failed/nonquiescent build')
    require(plan['configuration_id']==bundle['models'][plan['model']]['configuration_id'],'Selected CID')
    result=bound({'path':plan['selected_result'],'sha256':plan['reference_result_sha256']})
    config=bound({'path':plan['selected_config'],'sha256':plan['raw_config_sha256']})
    require(obj(config)['configuration_id']==plan['configuration_id']==obj(result)['configuration_id'],'Actual selected config/result join')
    args=plan['ncu_arguments'];value=lambda flag:args[args.index(flag)+1]
    for k,v in {'--profile-from-start':'off','--target-processes':'application-only','--devices':'1','--filter-mode':'global','--launch-count':'1','--kill':'0','--check-exit-code':'1','--clock-control':'none','--cache-control':'all','--replay-mode':'kernel','--print-units':'base','--page':'raw'}.items():require(value(k)==v,'NCU reviewed option '+k)
    require(set(value('--metrics').split(','))==set(METRICS+PCI+CC),'All27 standard counters plus3 PCI and2 CC required, no substitution')
    require(len(value('--metrics').split(','))==32,'Duplicate requested metric')
    require(str(Path(plan['ncu_executable']).resolve())==review['ncu_executable']['path'],'Exact old2022 executable, no2025 fallback')
    require(bundle['all_readonly_bindings'][plan['ncu_executable']]==review['ncu_executable']['sha256'],'Prospectively bound NCU identity')
    require(plan['environment_overrides']['CUDA_VISIBLE_DEVICES']=='5,6','Two-device mapping preserved')
    require(plan['execution_bindings']==bundle['models'][plan['model']]['execution_bindings'],'Execution binding map changed')
    require(prov['authorization']['proposal_sha256']==review['proposal']['sha256'] and prov['authorization']['runner_sha256']==review['runner']['sha256'] and prov['authorization']['execution_authorized']is True and prov['authorization']['authorization_origin']=='explicit_user_approval_recorded_by_root' and bool(prov['authorization']['user_authorization_reference']),'Explicit user GPU5 authorization receipt absent')
    require(prov['authorization']['physical_GPU_UUIDs']==bundle['physical_GPU_UUIDs'] and prov['authorization']['models']==list(MODELS),'User authorization exact devices/models')
    expected=bundle['physical_GPU_UUIDs'];require(expected['6']==UUID and expected==review['physical_GPU_UUIDs'],'Two physical UUIDs')
    for name in ('device_before','device_before_profile','device_after'):
        snap=prov[name];require({k:v['uuid']for k,v in snap['devices'].items()}==expected and not snap['compute_processes'],'Actual before/after both-device idle identity')
    require(pcie_tuple(prov['device_before_profile']['devices']['6']['pci'])==(0,203,0),'Physical GPU6 PCI mapping')
    registry=slot/'scratch/registry';ranks={r:obj(registry/f'rank{r}.json')for r in (0,1)};session=obj(registry/'session.json');profiler=obj(registry/'profiler.json')
    for r,x in ranks.items():
        require(x['rank']==r and x['uid']==session['uid'] and x['sid']==session['sid']==profile['session'] and x['token']==session['token'] and str(x['starttime']).isdigit() and x['target_argv']==plan['rank_target_argv'],'Captured rank process identity')
        env=x['effective_environment'];require(env['CUDA_VISIBLE_DEVICES']=='5,6' and env['OMPI_COMM_WORLD_SIZE']=='2' and env['OMPI_COMM_WORLD_RANK']==str(r) and env['OMPI_COMM_WORLD_LOCAL_RANK']==str(r),'Exact rank/visibility mapping')
    require(ranks[0]['pid']!=ranks[1]['pid'] and profiler['pid']not in [ranks[0]['pid'],ranks[1]['pid']],'Distinct actual rank/profiler PIDs')
    require(profiler['role']=='profiler' and profiler['rank']==1 and profiler['sid']==session['sid'] and profiler['token']==session['token'] and profiler['executable']==plan['ncu_executable'],'Captured profiler registration')
    attest=labelled((slot/'profile.log').read_text(errors='replace'),'ALLREDUCE_DEVICE_ATTEST ')
    require(len(attest)==4,'Both ranks runtime device attestations before warmup and after all5 checks')
    for r in (0,1):
        for phase in ('before_first_warmup_add','after_all5_full_checks'):
            matches=[x for x in attest if x.get('rank')==r and x.get('phase')==phase]
            require(len(matches)==1,'Unique runtime rank/phase attestation')
            x=matches[0];physical=str(r+5)
            require(x['logical_device']==r and x['pid']==ranks[r]['pid'] and x['uuid']==expected[physical] and pcie_tuple(x['pci'])==pcie_tuple(prov['device_before_profile']['devices'][physical]['pci']),'Actual both-rank PID/logicaldevice/UUID/PCI join')
            env=ranks[r]['effective_environment']
            for ph in ('5','6'):
                require(env['STUDY_EXPECTED_GPU'+ph+'_UUID']==expected[ph] and pcie_tuple(env['STUDY_EXPECTED_GPU'+ph+'_PCI'])==pcie_tuple(prov['device_before_profile']['devices'][ph]['pci']),'Registered expected physical device mapping')
    allowed={x['pid']:expected[str(r+5)]for r,x in ranks.items()};allowed[profiler['pid']]=UUID
    for snap in profile['GPU_monitor_snapshots']:
        require({k:v['uuid']for k,v in snap['devices'].items()}==expected,'Continuous UUID mapping')
        for app in snap['compute_processes']:require(allowed.get(app['pid'])==app['gpu_uuid'],'Observed unknown/wrong-rank/device GPU process')
    require(profile['GPU_monitor_snapshots'],'No monitored interval evidence')
    require(profile['leader_identity']['sid']==profile['session'] and str(profile['leader_identity']['starttime']).isdigit(),'Owned leader SID/start identity')
    log=(slot/'profile.log').read_text(errors='replace');trials=labelled(log,'STUDY_TRIAL ');begins=labelled(log,'ALLREDUCE_PROFILE_BEGIN ');ends=labelled(log,'ALLREDUCE_PROFILE_END ');done=labelled(log,'ALLREDUCE_PROFILE_DONE ')
    require([x['trial']for x in trials]==[1,2,3,4,5],'Actual five checked rows in original order')
    artifacts=[];ref=obj(result)['models']
    for t in trials:
        require(t['correct']is True and t['checked_values']==COUNT and t['jit_excluded']is True and t['internal_repeats']==18 and t['timing_region']=='sum6_max2rank_preallocated_source_ring_MPI_context_fenced_v2','Original five-trial contract not preserved')
        require(isinstance(t['time_seconds'],(int,float))and math.isfinite(t['time_seconds'])and t['time_seconds']>0,'Invalid profiled-run elapsed diagnostic')
        decl=t['output_artifact'];require(decl['dtype']=='float32' and decl['count']==COUNT and decl['byte_order']=='little','Full artifact representation')
        p=slot/'scratch'/f"trial{t['trial']}.bin";require(Path(decl['path']).name==p.name,'Actual full artifact path')
        a=exact_artifact(p)
        for refmodel in ('cuda','cuda_jl'):
            old=ref[refmodel]['trials'][t['trial']-1]['output_artifact'];oldpath=bound({'path':old['saved_path'],'sha256':old['sha256']})
            require(oldpath.stat().st_size==COUNT*4 and a['sha256']==old['sha256'],'Full bitwise dual-reference join')
        artifacts.append(a)
    require(len(begins)==len(ends)==1 and len(done)==2,'Exact single marker and both-rank completion')
    b,e=begins[0],ends[0]
    for k,v in {'rank':1,'trial':2,'n':1048576,'repeat':1,'offset':0,'count':524288,'pid':ranks[1]['pid']}.items():require(b.get(k)==v,'Begin marker '+k)
    for k,v in {'rank':1,'trial':2,'n':1048576,'repeat':1,'authored_add_dispatches':1,'pid':ranks[1]['pid']}.items():require(e.get(k)==v,'End marker '+k)
    require(sorted(x['rank']for x in done)==[0,1],'Both rank final guards')
    for x in done:
        r=x['rank'];require((x['pid'],x['ring_add_calls'],x['candidates'],x['enters'],x['exits'])==(ranks[r]['pid'],96,3 if r else 0,1 if r else 0,1 if r else 0),'Full six-case controller dispatch guards')
    groups,digest=parser.read_csv(slot/'counters.csv');require(len(groups)==1,'Exactly one rank1 add sample required')
    g=groups[0];require(set(METRICS+PCI+CC)<=g['metrics'].keys(),'Missing required metric row')
    require(all(isinstance(g['metrics'][k],(int,float)) and math.isfinite(g['metrics'][k]) and g['metrics'][k]>=0 for k in METRICS+PCI+CC),'Required counter nonnumeric/nonfinite')
    i=g['identity'];require(int(i['Process ID'])==ranks[1]['pid'] and re.search(plan['kernel_regex'],i['Kernel Name']),'Actual sample rank/kernel family')
    for k,v in [('Grid Size',[32,1,1]),('Block Size',[256,1,1])]:require([int(x)for x in re.findall(r'\d+',i[k])]==v,'Exact sampled shape')
    require(tuple(g['metrics'][k]for k in PCI)==(0,203,0),'Actual raw sample GPU6 PCI')
    require(tuple(g['metrics'][k]for k in CC)==(7,0),'Actual sample compute capability sm70')
    require(g['metrics']['gpu__time_duration.sum']>0,'Positive sample duration')
    require(g['metrics']['sm__inst_executed_pipe_tensor.sum']==0,'Unexpected tensor arithmetic requires separately reviewed typed mode; no implicit conversion')
    return {'trials':trials,'artifacts':artifacts,'begin':b,'end':e,'done':done,'rank_registry':ranks,'session_registry':session,'profiler_registry':profiler,'device_attestations':attest},groups,digest

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-sha256',required=True);ap.add_argument('--capture-root',type=Path,required=True);ap.add_argument('--output-root',type=Path,required=True);ap.add_argument('--execute-cpu-adapter',action='store_true');a=ap.parse_args()
    require(a.execute_cpu_adapter,'Source-only proposal; explicit root invocation needed')
    require(sha(a.review)==a.review_sha256,'Review SHA');review=obj(a.review)
    require(review['method_id']==METHOD and review['root_reviewed_for_cpu_adaptation']is True,'Root CPU method adoption review absent')
    require(review['adapter_sha256']==sha(Path(__file__)),'Exact adapter source')
    bundlepath=bound(review['proposal']);bundle=obj(bundlepath);bound(review['runner']);parserpath=bound(review['parser_source']);calpath=bound(review['calibration_review']);bound(review['collector_reference_source']);bound(review['ncu_executable'])
    require(bundle['runner']['sha256']==review['runner']['sha256'],'Runner prospective binding')
    for key in ('parser_source','calibration_review','collector_reference_source','ncu_executable'):
        item=review[key];require(bundle['all_readonly_bindings'].get(item['path'])==item['sha256'],'Missing prospective runner binding: '+key)
    require('2022.4.1' in review['profiler_version'],'Only old2022 calibrated stratum allowed')
    calibration=obj(calpath);require(calibration['device_uuid']==UUID and calibration['architecture']=='sm_70' and '2022.4.1'in calibration['tool_versions']['ncu'],'Exact old2022GPU6 calibration proof')
    for item in calibration['files']:
        p=Path(review['source_root'])/item['path'];require(sha(p)==item['sha256'],'Calibration supporting file changed');require(bundle['all_readonly_bindings'].get(str(p))==item['sha256'],'Missing prospective calibration supporting binding')
    raw=a.capture_root.resolve();out=a.output_root.resolve();require(raw.is_dir()and out.is_relative_to('/tmp')and not out.exists()and not out.is_symlink()and not raw.is_relative_to(out)and not out.is_relative_to(raw),'Fresh disjoint /tmp output required')
    require(all((raw/'allreduce'/m/'profile_provenance.json').is_file()for m in MODELS),'Require all four completed or explicit failed model casefiles; pending is not zero')
    # Capture immutable raw bytes before any conversion; never write into them.
    initial=archive_tree(raw);out.mkdir(parents=True,mode=0o700);dump(out/'original_raw_inventory.json',initial)
    parser=types.ModuleType('reviewed_retained_parser');parser.__file__=str(parserpath);exec(compile(parserpath.read_bytes(),str(parserpath),'exec'),parser.__dict__)
    outcomes=[]
    for model in MODELS:
        source=raw/'allreduce'/model;dest=out/'allreduce'/model;dest.mkdir(parents=True)
        origplan=obj(source/'plan.json');parent=obj(source/'profile_provenance.json');plan=dict(origplan)
        require(origplan['model']==model and parent['model']==model,'Model directory join')
        plan.update(benchmark='allreduce',model=model,reference_result=origplan['selected_result'],profile_method=METHOD,profile_capture_parent=str(source),profile_method_review={'path':str(a.review.resolve()),'sha256':a.review_sha256},staged=True)
        dump(dest/'plan.json',plan)
        # The collector_source is the ACTUAL custom process controller, never
        # falsely labeled as ordinary v100_roofline_collect.py.
        (dest/'collector_source.py').write_bytes(bound(review['runner']).read_bytes());(dest/'parser_source.py').write_bytes(parserpath.read_bytes());(dest/'adapter_source.py').write_bytes(Path(__file__).read_bytes());(dest/'parent_profile_provenance.json').write_bytes((source/'profile_provenance.json').read_bytes())
        for name in ('counters.csv','profile.log'):
            if(source/name).is_file():(dest/('execution.log'if name=='profile.log'else name)).write_bytes((source/name).read_bytes())
        prov={'schema_version':1,'started_at':parent.get('started_at'),'finished_at':parent.get('finished_at'),'status':'custom_capture_adapter_validation_failed','gpu_uuid':UUID,'collector_sha256':review['runner']['sha256'],'plan_sha256':sha(dest/'plan.json'),'plan':str(dest/'plan.json'),'selector_scope':'Separate two-device rank1 private add; retained trial2 n1M firstrepeat only, not collective or complete ROI','ROI_coverage':'single_source_marked_rank1_add_not_collective_ROI','primary_contract_unchanged':True,'primary_timing_replacement':False,'replay_mode':'kernel','filesystem_write_guard':bundle['write_guard'],'allreduce_two_device_method':{'method_id':METHOD,'review_sha256':a.review_sha256,'parent_provenance_sha256':sha(source/'profile_provenance.json'),'parent_plan_sha256':sha(source/'plan.json'),'adapter_sha256':sha(Path(__file__)),'runtime_checks':None},'command':parent.get('profile',{}).get('command'),'environment_overrides':origplan.get('environment_overrides'),'profile':parent.get('profile'),'builds':parent.get('builds',[]),'new_gpu_processes':0,'new_build_processes':0,'CPU_capture_adaptation':True,'binding_recheck_after':None,'source_identity_basis':'Captured runner proposal/execution bindings verified before and after its actual execution; no historical source identity reconstructed from current files.'}
        if(dest/'counters.csv').is_file():prov['counters_sha256']=sha(dest/'counters.csv')
        try:
            checks,groups,digest=validate_slot(source,origplan,parent,bundle,review,parser)
            calibration_link={'path':str(calpath),'sha256':review['calibration_review']['sha256'],'half_lane_factor':2,'default_tensor_conversion_enabled':False,'reason':'Exact source-bound oldNCU2022 GPU6 physical packed-lane convention; no2025 fallback.'}
            context={k:plan[k]for k in ('benchmark','model','configuration_id','reference_result_sha256','raw_config_sha256')}
            context.update(profile_method=METHOD,gpu_sm70=True,retained_trial=2,sample_phase='source_marked_rank1_trial2_size1M_firstrepeat_private_add',sample_in_primary_timed_invocation=True,first_invocation_may_be_warmup=False,semantic_family_in_primary_ROI=True,ROI_stage='rank1_local_add_only',ROI_coverage='single_rank_add_not_collective_ROI',clock_control='none',cache_policy='all_cold_replay',replay_mode='kernel',source_context_bound_before_after=True,communication_replayed=False,physical_device_uuid=UUID,rank=1,rank_count=2,whole_ROI_claim=False,fp16_work_semantics='Physical executed packed lanes, including unused scalarhalf high lanes; distinct from useful work.')
            records=[parser.derive(g,half_lanes=2,half_provenance=calibration_link,context=context)for g in groups]
            require(all(not x['warnings']for x in records),'Unknown units/incomplete math: retain counters but do not publish positions')
            dump(dest/'derived.json',{'schema_version':1,'input_csv':str(dest/'counters.csv'),'input_sha256':digest,'parser_sha256':review['parser_source']['sha256'],'kernel_count':len(records),'records':records,'raw_metrics':groups,'benchmark_bottleneck':'not_aggregated'})
            prov.update(status='profile_process_completed',kernel_count=1,profile_device_identity_verified='Every sampled raw row physicalPCI0:203:0 matches contemporaneousGPU6UUID and rank1PID',expected_device_pci={'domain':0,'bus':203,'device':0},additional_current_profile_input_bindings=bundle['all_readonly_bindings'],binding_recheck_after='unchanged',operation_calibration=calibration_link)
            prov['allreduce_two_device_method']['runtime_checks']=checks
        except Exception as error:prov['adapter_validation_error']=repr(error)
        # Compact complete method evidence archive; raw large artifacts remain
        # individually hash-bound at original paths, never discarded or modified.
        small=[]
        for p in sorted(source.rglob('*')):
            if p.is_file()and p.suffix in ('.json','.log')and p.stat().st_size<=16*1024**2:
                b=p.read_bytes();small.append({'path':str(p.relative_to(source)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'base64_bytes':base64.b64encode(b).decode()})
        envelope={'schema_version':1,'method_id':METHOD,'review':review,'parent_plan':origplan,'parent_plan_sha256':sha(source/'plan.json'),'parent_provenance_sha256':sha(source/'profile_provenance.json'),'raw_files':[x for x in initial if x['path'].startswith('allreduce/'+model+'/')],'compact_exact_raw_files':small,'adoption':'Requires explicit frozen-study audit branch and freeze allowlist; custom success is not automatically ordinary collector-compatible.'}
        dump(dest/'allreduce_two_device_review.json',envelope);prov['allreduce_two_device_method']['archive_sha256']=sha(dest/'allreduce_two_device_review.json');dump(dest/'profile_provenance.json',prov);outcomes.append({'model':model,'status':prov['status'],'error':prov.get('adapter_validation_error')})
    require(archive_tree(raw)==initial,'Parent evidence mutated during CPU adaptation; output must not be imported')
    dump(out/'adapter_receipt.json',{'schema_version':1,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'adapter_sha256':sha(Path(__file__)),'review_sha256':a.review_sha256,'raw_inventory_sha256':sha(out/'original_raw_inventory.json'),'parent_bytes_unchanged':True,'outcomes':outcomes,'GPU_runs':0,'build_runs':0,'adoption_required':True,'status':'ready_for_root_integration_review_not_automatically_published'})
    return 0 if all(x['status']=='profile_process_completed'for x in outcomes)else 1
if __name__=='__main__':raise SystemExit(main())
